import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import MetamerMismatch as mm

def mesh_from_hull(hull, name = None): #hull : ConvexHull
    vertices, faces = hull.points, hull.simplices
    
    legend_text = 'Metamer body colors'
    legendgroup = 'group1'
    color = None
	    
    if name == 'ocs':
        color, legend_text, legendgroup = 'black', 'Object Colour Solid', 'group0.5'
        
    mesh = \
        go.Mesh3d(
        x=np.array(vertices)[:,0], y=np.array(vertices)[:,1], z=np.array(vertices)[:,2],
        i=np.array(faces)[:,0]   , j=np.array(faces)[:,1]   , k=np.array(faces)[:,2],
        opacity=0.5, color = color, name = name,
        legendgrouptitle_text = legend_text,
        legendgroup=legendgroup) #,  alphahull = 0 # ?
    
    return mesh


def plot_metamer_body(mmbs, ocs = None): #ocs : ConvexHull = None
	#mmb : ConvexHull
    
	layout = go.Layout(scene = dict(
	            xaxis = dict(range = [0,1], title = 'r', showgrid = False),
	            yaxis = dict(range = [0,1], title = 'g', showgrid = False),
	            zaxis = dict(range = [0,1], title = 'b', showgrid = False),
	            aspectmode='cube'), legend_font_size = 20)

	meshes = []

	if ocs is not None:
		meshes.append(mesh_from_hull(ocs, 'ocs'))

	for mmb in mmbs:
		meshes.append(mesh_from_hull(mmb))

	fig = go.Figure(data = meshes, layout = layout)
	fig.update_traces(showlegend=True, showscale=False)

	return fig

def plot_scene(colors_mmb, ocs = None, color_predictions = None):
	fig = plot_metamer_body(mmbs = colors_mmb, ocs = ocs)

	if color_predictions is not None:
		for ind, color in enumerate(color_predictions):
			fig.add_trace(go.Scatter3d(x = [color[0]], 
									   y = [color[1]], 
									   z = [color[2]],
						mode = 'markers', name = f'pred color{ind + 1}',
			            legendgrouptitle_text = 'Predicted Colors',
			            legendgroup="group2"))

	fig.show()
	return fig

