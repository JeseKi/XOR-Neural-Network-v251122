from pathlib import Path
from typing import List

import plotly.graph_objects as go # type: ignore

from nn.schemas import History


def plot_loss(
    history: List[History], file_path: str | Path = Path("loss.html")
) -> go.Figure:
    epochs = [h.epoch for h in history]
    losses = [h.loss for h in history]

    fig = go.Figure(data=go.Scatter(x=epochs, y=losses, mode="lines+markers"))
    fig.update_layout(
        title="损失变化曲线",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
    )
    fig.write_html(file_path)

    return fig


def animate_weights(
    history: List[History], file_path: str | Path = Path("weights_animation.html")
) -> go.Figure:
    if not history:
        raise ValueError("历史记录不能为空")

    # 1. 定义节点位置
    node_positions = {
        "input_1": (0, 2),
        "input_2": (0, 1),
        "hidden_1": (1, 2.5),
        "hidden_2": (1, 0.5),
        "output": (2, 1.5),
    }

    # 绘制神经元节点
    node_trace = go.Scatter(
        x=[pos[0] for pos in node_positions.values()],
        y=[pos[1] for pos in node_positions.values()],
        text=list(node_positions.keys()),
        textposition="top center",
        mode="markers+text",
        marker=dict(size=20, color="lightblue"),
        name="神经元",
        hoverinfo="text",
    )

    # 2. 定义权重连接关系
    weight_connections = {
        "w_i1_h1": ("input_1", "hidden_1"),
        "w_i2_h1": ("input_2", "hidden_1"),
        "w_i1_h2": ("input_1", "hidden_2"),
        "w_i2_h2": ("input_2", "hidden_2"),
        "w_h1_o1": ("hidden_1", "output"),
        "w_h2_o1": ("hidden_2", "output"),
    }

    # 计算权重圆圈的位置
    weight_positions = {
        name: (
            (node_positions[start][0] + node_positions[end][0]) / 2,
            (node_positions[start][1] + node_positions[end][1]) / 2,
        )
        for name, (start, end) in weight_connections.items()
    }

    # 3. 为信号的动画定义不同的 dash 样式
    dash_styles = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]

    frames = []

    for i, item in enumerate(history):
        # 将权重收集到一个字典中
        current_weights = {
            "w_i1_h1": item.input_to_hidden_weights[0][0],
            "w_i2_h1": item.input_to_hidden_weights[1][0],
            "w_i1_h2": item.input_to_hidden_weights[0][1],
            "w_i2_h2": item.input_to_hidden_weights[1][1],
            "w_h1_o1": item.hidden_to_output_weights[0],
            "w_h2_o1": item.hidden_to_output_weights[1],
        }

        frame_traces = []

        for name, pos in weight_positions.items():
            weight_val = current_weights[name]
            font_size = min(16, 2 + abs(weight_val) * 10)  # 动态字体大小
            frame_traces.append(
                go.Scatter(
                    x=[pos[0]],
                    y=[pos[1]],
                    mode="markers+text",
                    text=[f"{name}<br>{weight_val:.2f}"],
                    textposition="middle center",
                    marker=dict(
                        color="orangered" if weight_val > 0 else "deepskyblue",
                        size=4 + abs(weight_val) * 30,
                    ),
                    name=name,
                    hoverinfo="text",
                    textfont=dict(size=font_size, color="white"),
                )
            )

        # b) 动态连接线
        # 确定当前帧的 dash 样式
        dash_style = dash_styles[i % len(dash_styles)]
        for name, (start_node, end_node) in weight_connections.items():
            frame_traces.append(
                go.Scatter(
                    x=[node_positions[start_node][0], node_positions[end_node][0]],
                    y=[node_positions[start_node][1], node_positions[end_node][1]],
                    mode="lines",
                    line=dict(color="lightgrey", width=1, dash=dash_style),
                    hoverinfo="none",  # 隐藏这条线的悬停信息
                )
            )

        frames.append(go.Frame(data=frame_traces, name=str(item.epoch)))

    # 创建初始布局（节点 + 第一帧的迹线）
    initial_traces = [node_trace] + list(frames[0].data) if frames else [node_trace]
    fig = go.Figure(data=initial_traces)

    # 设置动画
    fig.update(frames=frames)

    # 定义动画播放器
    fig.update_layout(
        title_text=f"Neural Network Weights Animation (Epoch {history[0].epoch if history else 'N/A'})",
        xaxis=dict(visible=False, range=[-0.5, 2.5]),
        yaxis=dict(visible=False, range=[0, 4]),
        showlegend=False,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "label": str(item.epoch),
                        "method": "animate",
                        "args": [
                            [str(item.epoch)],
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                    }
                    for item in history
                ],
                "active": 0,
                "transition": {"duration": 0},
                "x": 0.1,
                "len": 0.9,
            }
        ],
    )

    # 动态更新每一帧的标题
    for frame in fig.frames:
        frame.layout = go.Layout(  # type: ignore
            title_text=f"Neural Network Weights Animation (Epoch {frame.name})"  # type: ignore
        )

    fig.write_html(file_path)

    return fig
