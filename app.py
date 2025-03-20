import base64
import io
import os
import zipfile
import dash
from dash import dcc, html, Input, Output, State
import ffmpeg
import cv2
import numpy as np
import datetime
import plotly.graph_objects as go

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1(children='Video Parser', style={'textAlign': 'center'}),
    dcc.Upload(
        id='upload-video',
        children=html.Div(['Перетащите или выберите MP4-файл']),
        className="upload-box",
        style={
            'width': '70%',
            'height': '80px',
            'lineHeight': '80px',
            'textAlign': 'center',
        },
        multiple=False
    ),
    html.Div([
        html.Label("Установите величину FPS:"),
        dcc.Input(
            id='fps-input',
            type='number',
            value=5,
            placeholder='Введите FPS',
            style={'marginLeft': '10px', 'marginRight': '10px', 'font-weight': 'bold'}
        ),
    ], style={'marginBottom': '20px'}),
    html.Div([
        html.Label("Задайте промежуток времени в секундах:"),
        dcc.Input(
            id='start-time',
            type='number',
            placeholder='Начало (секунды)',
            style={'marginLeft': '10px', 'marginRight': '10px', 'font-weight': 'bold'}
        ),
        dcc.Input(
            id='end-time',
            type='number',
            placeholder='Конец (секунды)',
            style={'marginLeft': '10px', 'marginRight': '10px', 'font-weight': 'bold'}
        ),
    ], style={'marginBottom': '20px'}),
    html.Button("Выбрать 4 точки", id="quad-button", style={'margin': '10px'}),
    html.Button("Получить фреймы", id="process-button", style={'margin': '10px'}),  
    html.Button("Скачать фреймы", id="download-button", style={'margin': '10px'}),
    dcc.Download(id="download-zip"),
    html.Div(id='output-message'),
    html.Div(id='first-frame-container', style={'textAlign': 'center', 'margin': '20px'}),
    dcc.Graph(id='first-frame-graph', style={'display': 'none'}), 
    dcc.Store(id='points-store')
])

def convert_frame_to_base64(frame_path):
    with open(frame_path, "rb") as img_file:
        return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"

@app.callback(
    Output('first-frame-container', 'children'),
    [Input('quad-button', 'n_clicks')],
    [State('upload-video', 'contents')],
    prevent_initial_call=True
)
def show_first_frame(n_clicks, contents):
    if n_clicks is None or contents is None:
        return html.Div()

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    temp_video_path = "temp.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(decoded)

    cap = cv2.VideoCapture(temp_video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        return html.Div("Не удалось извлечь первый фрейм.")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    os.remove(temp_video_path)

    fig = go.Figure()
    fig.add_trace(go.Image(z=frame_rgb))
    fig.update_layout(
        dragmode='select',  
        margin=dict(l=0, r=0, t=0, b=0),
        shapes=[],  
        newshape=dict(line=dict(color="red", width=2))  
    )

    return html.Div([
        dcc.Graph(
            id='first-frame-graph',
            figure=fig,
            style={'width': '50%', 'cursor': 'crosshair'}
        ),
        html.Div("Выберите 4 точки на изображении.", style={'marginTop': '10px'})
    ])

@app.callback(
    Output('points-store', 'data'),
    [Input('first-frame-graph', 'clickData')],
    [State('points-store', 'data')],
    prevent_initial_call=True
)
def store_points(clickData, points):
    if clickData is None:
        return points or []
    points = points or []
    x = clickData['points'][0]['x']
    y = clickData['points'][0]['y']
    points.append((x, y))
    return points  # Ограничиваем до 4 точек

@app.callback(
    Output('first-frame-graph', 'figure'),
    [Input('points-store', 'data')],
    [State('first-frame-graph', 'figure')],
    prevent_initial_call=True
)
def update_graph(points, figure):
    if not points:
        return figure

    figure['layout']['shapes'] = []

    if len(points) > 1:
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            figure['layout']['shapes'].append(
                dict(
                    type="line",
                    x0=x0, y0=y0,
                    x1=x1, y1=y1,
                    line=dict(color="red", width=2)
                )
            )
    if len(points) == 4:
        x0, y0 = points[-1]
        x1, y1 = points[0]
        figure['layout']['shapes'].append(
            dict(
                type="line",
                x0=x0, y0=y0,
                x1=x1, y1=y1,
                line=dict(color="red", width=2)
            )
        )

    return figure

def sort_points_clockwise(points):
    points = np.array(points, dtype=np.float32)

    center = np.mean(points, axis=0)

    def get_quadrant(point):
        x, y = point
        cx, cy = center
        if x < cx and y >= cy:
            return 1  
        elif x < cx and y < cy:
            return 2  
        elif x >= cx and y < cy:
            return 3 
        else:
            return 4 
    sorted_points = sorted(points, key=lambda point: get_quadrant(point))

    return sorted_points

@app.callback(
    Output('output-message', 'children'),
    [Input('process-button', 'n_clicks')],
    [
        State('upload-video', 'contents'),
        State('fps-input', 'value'),
        State('start-time', 'value'),
        State('end-time', 'value'),
        State('points-store', 'data')
    ],
    prevent_initial_call=True
)

def process_frames(n_clicks, contents, fps, start_time, end_time, points):
    if n_clicks is None or contents is None:
        return html.Div(["Загрузите видео и нажмите 'Получить фреймы'."])

    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = float('inf')
    if fps is None or fps <= 0:
        fps = 5  
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    temp_video_path = "temp.mp4"
    frames_raw_dir = "frames_raw" 
    frames_processed_dir = "frames_processed"  
    os.makedirs(frames_raw_dir, exist_ok=True)
    os.makedirs(frames_processed_dir, exist_ok=True)

    try:
        with open(temp_video_path, "wb") as f:
            f.write(decoded)

        probe = ffmpeg.probe(temp_video_path)
        duration = float(probe['format']['duration'])

        start_time = max(0, start_time)
        end_time = min(duration, end_time) if end_time != float('inf') else duration

        raw_frame_pattern = os.path.join(frames_raw_dir, "frame_%04d.png")
        (
            ffmpeg
            .input(temp_video_path, ss=start_time, t=end_time - start_time)
            .filter("fps", fps=fps)
            .output(raw_frame_pattern)
            .run(capture_stdout=True, capture_stderr=True)
        )

        raw_frame_paths = sorted([os.path.join(frames_raw_dir, f) for f in os.listdir(frames_raw_dir)])
        if not raw_frame_paths:
            return html.Div("Не удалось извлечь фреймы.")

        first_frame = cv2.imread(raw_frame_paths[0])
        h, w = first_frame.shape[:2]

        if points is not None:
            points = points[:4]
            points_src = sort_points_clockwise(points)
            points_dst = np.array([[0, h], [0, 0], [w, 0], [w, h]], dtype=np.float32)
            points_src = np.array(points_src, dtype=np.float32)
            H, _ = cv2.findHomography(points_src, points_dst)

            for i, frame_path in enumerate(raw_frame_paths):
                frame = cv2.imread(frame_path)
                frame_warped = cv2.warpPerspective(frame, H, (w, h))
                processed_frame_path = os.path.join(frames_processed_dir, f"frame_{i:04d}.png")
                cv2.imwrite(processed_frame_path, frame_warped)

            frame_paths = sorted([os.path.join(frames_processed_dir, f) for f in os.listdir(frames_processed_dir)])
        else:
            frame_paths = raw_frame_paths

        frames = [convert_frame_to_base64(frame_path) for frame_path in frame_paths]

        def safe_remove(file_path):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except PermissionError:
                print(f"Не удалось удалить файл {file_path}, так как он занят.")

        safe_remove(temp_video_path)

        return html.Div([
            html.H5("Фреймы успешно извлечены!"),
            html.Div([
                html.Img(src=frame, style={'width': '200px', 'margin': '5px'}) for frame in frames[:10]
            ])
        ])

    except ffmpeg.Error as e:
        return html.Div([f"Ошибка FFmpeg: {e.stderr.decode('utf-8')}"])
    
@app.callback(
    Output("download-zip", "data"),
    [Input("download-button", "n_clicks")],
    [
        State("fps-input", "value"),
        State("start-time", "value"),
        State("end-time", "value")
    ],
    prevent_initial_call=True
)
def download_frames(n_clicks, fps, start_time, end_time):
    if n_clicks is None:
        return None

    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = float('inf')

    frames_dir = "frames_processed" 
    if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
        frames_dir = "frames_raw"
        if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
            return html.Div(["Сначала нажмите 'Получить фреймы'."])

    frame_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)])

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for frame_path in frame_paths:
            zipf.write(frame_path, os.path.basename(frame_path))

    def safe_remove(file_path):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except PermissionError:
            print(f"Не удалось удалить файл {file_path}, так как он занят.")

    for frame_path in frame_paths:
        safe_remove(frame_path)
    if os.path.exists(frames_dir):
        os.rmdir(frames_dir)

    zip_buffer.seek(0)
    return dcc.send_bytes(zip_buffer.read(), filename=f"frames_{datetime.datetime.now().replace(microsecond=0)}.zip")

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8051)