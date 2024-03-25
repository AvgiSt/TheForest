import argparse
import dash
from dash import Dash, html, dcc, ClientsideFunction
from dash.dependencies import Input, Output, State, ALL
from dash_extensions import WebSocket 
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorsys
import json

class GUI():

    def __init__(self, ws_host, ws_port):

        self.app = Dash(
            __name__, 
            external_stylesheets=[dbc.themes.QUARTZ],
            update_title=None,
            # prevent_initial_callbacks=True,
        )

        self.ws = (ws_host, ws_port)

        self.pillar_figure = go.Figure()

        self.pillar_status_children = None
        self.data = None

        self.update_dat = dict( # ["amp", "note-pitch", "synth", "bpm", "pan", "envelope"]
            serial_port={},
            amp = {},
            pitch = {},
            synth = {},
            bpm = {},
            pan = {},
            envelope = {}
        )

        self.display_dat = dict(
            touch = {},
            params = {}
        )

        self.app.layout = self.init_layout
        self.init_callbacks()


    def init_layout(self):
        ###############
        ## Layouts ##
        ###############

        self.pillar_graph = html.Div([
            dcc.Graph(id="pillar_graph", figure=self.pillar_figure)
        ])

        self.system_status = html.Div([
            html.Div(id='output', style={'display': 'none'}),
            # This goes under pillar graph
            #dbc.Row([
            #    dbc.Label("Current Beats Per Minute", width=2),
            #    dbc.Label(id="bpm-output", width=1),
            #    dbc.Col(dbc.Input(id="bpm-input", type="number", placeholder="Enter BPM"), width=4),
            #    dbc.Col(dbc.Button("Update", id="bpm-button", color="primary"), width=2),
            #]),
    
            html.Div(id='system-status-dummy-output')
        ])

        self.pillar_status = html.Div(id="pillar-status", children=[]) # Fill in dynamically

        self.html_main_content = html.Div(className='container-fluid', children=[
            html.Div(className='row', children=[
                self.system_status,   
                self.pillar_graph,
                self.pillar_status
            ])
            ], style={
                'padding': '10px 5px',
            })
        

        self.utility_content = html.Div([
            dcc.Interval(id='interval-component', interval=1000/5, n_intervals=0),  # Trigger every 200ms
            html.Div(id="websocket-holder", children=WebSocket(id="ws", url=f"ws://{self.ws[0]}:{self.ws[1]}")),
            html.Div(id='dummy-output', style={'display': 'none'}),  # Hidden dummy output component
            html.Div(id={"type": f"pillar-status-label", "index": "dummy"}, style={'display':'none'})
        ])

        self.html_header_content = html.Div([
                dbc.Row([
                    dbc.Col(html.Div(
                    html.H1('Alien-Forest Dashboard'),
                    style=dict(padding=5)
                    ), width=6),
                    # dbc.Col(dbc.Input(id="websocket-input"), width=4),
                    dbc.Col(dbc.Button("Refresh", id="refresh-button"), width="auto")
                    ], justify="end"
                ),
                self.utility_content,
            ], style={
                'height': '100%',
                'overflow': "hidden",
                'borderBottom': 'thin lightgrey solid',
        })

        return html.Div([
            self.html_header_content,
            self.html_main_content
        ])


    def init_callbacks(self):
        ###############
        ## CALLBACKS ##
        ###############

        # self.app.clientside_callback(
        #     ClientsideFunction(
        #         namespace="clientside",
        #         function_name='read_message'
        #     ), 
        #     Output("output", "children"), 
        #     Input("ws", "message"))

        @self.app.callback(
            Output("pillar-status", "children"),
            Input("refresh-button", "n_clicks")
        )
        def refresh_page(n_clicks):
            if self.data:
                return self.generate_pillar_status(self.data)
            return dash.no_update
        
        @self.app.callback(
            [
                Output("output", "children"),
                Output("pillar_graph", "figure"),
                #Output("bpm-output", "children"),
                Output({"type": f"pillar-status-label", "index": ALL}, "children") 
            ], 
            [Input("ws", "message")])
        def receive_from_websocket(e):
            if e is None:
                                
                return [dash.no_update for _ in range(5)] + [ [dash.no_update] for _ in dash.callback_context.outputs_list[2] ]    # This is [touch, notes, serial_port, amp, synth] should change to [serial_port, touch,note_pitch,amp,synth,bpm,pan,envelope]

            data = json.loads(e['data']) # This is the data from the websocket?
            print(e)
            self.data = data
            #print(f"Received data: {data}")

            self.update_data(data)

            self.pillar_figure = self.generate_pillars_figure(data)

            outputs = dash.callback_context.outputs_list
            status_labels_update = outputs[7]
            labs = self.generate_updated_labels(status_labels_update)

            return [
                f"Response from websocket: {data}", 
                self.pillar_figure,
                #data["bpm"],
                labs
            ]

        @self.app.callback(
            [
                Output("ws", "send"),
                Output("system-status-dummy-output", "children"),
            ],
            [
                #Input("bpm-button", "n_clicks"),
                Input({"type": f"pillar-status-button", "index": ALL}, "n_clicks"),
                Input("pillar_graph", "clickData")
            ],
            [
                #State("bpm-input", "value"),
                State({"type": f"pillar-status-input", "index": ALL}, "value")
            ]
        )
        def update_output(graph_click_data, psi_value):
            ctx = dash.callback_context
            msg = dash.no_update
            ws_out = {}
            if ctx.triggered:
                triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

                
                if triggered_id == "pillar_graph":
                    data = graph_click_data["points"]
                    if len(data) > 0:
                        point = data[0]
                        p_id = point["curveNumber"]
                        t_id = point["pointNumber"]
                        out = [False for _ in range(int(self.data["pillars"][str(p_id)]["num_tubes"]))]
                        out[t_id] = True
                    else:
                        out = [False for _ in range(int(self.data["pillars"][str(0)]["num_tubes"]))]
                    if "touch" not in ws_out:
                        ws_out["touch"]  = {}
                    ws_out["touch"][str(p_id)] = out
                elif "{" in triggered_id and any(psi_value):
                    # One of the dynamic ones triggered lets find which one
                    prop_dict = json.loads(triggered_id)

                    l = [s["id"]['index'] for s in ctx.states_list[2]]
                    psi_value_idx = l.index(prop_dict["index"])
                    
                    p_id, var = prop_dict["index"].split("-")
                    if var not in ws_out:
                        ws_out[var] = {}
                    ws_out[var][p_id] = psi_value[psi_value_idx]
                    
            if ws_out:
                print(f"Sending {ws_out}")
            return [
                json.dumps(ws_out),
                msg
            ]

    def generate_pillars_figure(self, data):
        pillars_dict, current_status = data['pillars'], data["current_state"]
        num_pillars = len(pillars_dict)

        def generate_coords_tubes(num_tubes, radius):

            # Calculate the angles at which the points should be placed
            angles = np.linspace(0, 2*np.pi, num_tubes, endpoint=False)
            
            # Calculate the x and y coordinates of the points on the circle
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            
            points = [[x[i], y[i]] for i in range(num_tubes)]

            # Move (0,0) to front if central tube is first in array
            return np.array([[0,0]] + points)
        
        dist_between_centers = np.array([0, 100])
        
        fig = make_subplots(rows=1, cols=num_pillars, 
                            subplot_titles=[f"Pillar {p_id}" for p_id in pillars_dict])

        for i, (p_id, pillar) in enumerate(pillars_dict.items()):
            tube_coords = generate_coords_tubes(int(pillar["num_sensors"]), 1)
            current_status_lights = current_status[str(i)]["lights"]
            colours = [f"hsv({l[0]}, 255, {l[1]})" for l in current_status_lights]
            marker_widths = [10 if bool(t) else 2 for t in pillar["touch_status"]]
            marker_colors = ["DarkSlateGrey" if bool(t) else "white" for t in pillar["touch_status"]]
            labels = [f"{i},hue:{l[0]}" for i, l in zip(range(pillar["num_tubes"]), current_status_lights)]
            trace = go.Scatter(
                x=tube_coords[:, 0], 
                y=tube_coords[:, 1],
                marker=dict(
                    color=colours, 
                    size=50,
                    line=dict(
                        color=marker_colors,
                        width=marker_widths
                    )
                ),
                mode="markers+text",
                text=labels,
                textposition="top center")
            
            fig.add_trace(trace, row=1, col=i+1)
        
        fig.update_layout(showlegend=False)
        fig.update_layout(clickmode='event')
            
        return fig


    def update_data(self, data):
        num_pillars = data["num_pillars"]
        for p_id in range(num_pillars):
            self.update_dat["serial_port"][p_id]=data["pillars"][str(p_id)]["serial_status"]["port"]
            #["amp", "note-pitch", "synth", "bpm", "pan", "envelope"]
            self.update_dat["amp"][p_id] = data["amp"][str(p_id)]
            self.update_dat["pitch"][p_id] = data["pitch"][str(p_id)]
            self.update_dat["synth"][p_id] = data["synths"][str(p_id)]
            self.update_dat["bpm"][p_id] = data["bpm"][str(p_id)]
            self.update_dat["pan"][p_id] = data["pan"][str(p_id)]
            self.update_dat["envelope"][p_id] = data["envelope"][str(p_id)]
            

            self.display_dat["touch"][p_id] = data["pillars"][str(p_id)]["touch_status"]  # display what? where?
            self.display_dat["params"][p_id] = data["params"][str(p_id)]
            #self.display_dat["pan"][p_id] = data["pan"][str(p_id)]

    def generate_pillar_status(self, data):
        num_pillars = data["num_pillars"]

        output_div = []
        for p_id in range(num_pillars):
            
            update_dat = {n: d[p_id] for n, d in self.update_dat.items() if p_id in d}

            display_dat = {n: d[p_id] for n, d in self.display_dat.items() if p_id in d}

            connected = bool(data["pillars"][str(p_id)]["serial_status"]["connected"])
            row_div = [
                html.Div(children=f"Pillar {p_id} status: {'connected' if connected else 'not-connected'}"),
            ]
            for n, val in display_dat.items():
                h = dbc.Col(dbc.Row([
                    dbc.Label(f"{n}", width=2),
                    dbc.Label(id={"type": f"pillar-status-label", "index": f"{p_id}-{n}"}, width="auto"),
                ]))
                row_div.append(h)

            for n, val in update_dat.items():
                h = dbc.Row([
                    dbc.Label(f"{n}", width=2),
                    dbc.Label(id={"type": f"pillar-status-label", "index": f"{p_id}-{n}"}, width=2),
                    dbc.Col(dbc.Input(id={"type": f"pillar-status-input", "index": f"{p_id}-{n}"}, placeholder=f"Enter {n}"),width="auto"),
                    dbc.Col(dbc.Button("Update", id={"type": f"pillar-status-button", "index": f"{p_id}-{n}"}, color="primary"),width=1),
                ])
                row_div.append(h)
            output_div.append(dbc.Col(html.Div(row_div), width="auto"))
    
        return dbc.Row(output_div, justify="evenly")
    
    def generate_updated_labels(self, label_objs):

        status_labels = []
        for out_id in label_objs:
            name = out_id['id']['index']
            if name == "dummy":
                status_labels.append(dash.no_update)
                continue
            p_id, var = name.split("-")
            if var in self.update_dat:
                value = self.update_dat[var][int(p_id)]
            elif var in self.display_dat:
                value = self.display_dat[var][int(p_id)]
            else:
                status_labels.append(dash.no_update)
            status_labels.append(f"{value}")
        return status_labels
        

    def run(self, **kwargs):
        self.app.run(**kwargs)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="A script to parse host, port, and config file path.")
    parser.add_argument("--host", default="127.0.0.1", help="The host to connect to.")
    parser.add_argument("--port", default="8080", type=int, help="The port to use.")
    parser.add_argument("--debug", default=False, action="store_true", help="Whether to run the Dash GUI")

    parser.add_argument("--ws-host", default="localhost", help="The internal websocket URI")
    parser.add_argument("--ws-port", default="8765", help="The internal websocket URI")

    args = parser.parse_args()

    GUI(args.ws_host, args.ws_port).run(debug=True, host=args.host, port=args.port)


