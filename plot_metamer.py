import numpy as np
import plotly.graph_objs as go
from colorutils import rgb_to_string

def mesh_from_hull(hull, name = None, title_text = None, color = None):
    vertices, faces = hull.points, hull.simplices

    mesh = \
        go.Mesh3d(
        x=np.array(vertices)[:,0], y=np.array(vertices)[:,1], z=np.array(vertices)[:,2],
        i=np.array(faces)[:,0]   , j=np.array(faces)[:,1]   , k=np.array(faces)[:,2],
        opacity=0.5, color = color, name = name,
        legendgrouptitle_text = title_text,
        legendgroup = title_text) #,  alphahull = 0 # ?
    
    return mesh

def plot_scene(colors_mmb, color_predictions = None, ocs = None):
	"""
    plots scene 

    Parameters
    ----------
    colors_mmb : array_like
        array where each element is type ConvexHull
    color_predictions : (M, 3) array_like, optional
		scatter plots colors if not 'None'
	ocs : ConvexHull, optional
		plots object colour solid if not 'None' 

    Returns
    -------
    fig : class Figure
        returns go.Figure that can be used to add/update currently existing plot.
    """
	layout = go.Layout(scene = dict(
	            xaxis = dict(range = [0,1], title = 'RED', showgrid = False),
	            yaxis = dict(range = [0,1], title = 'GREEN', showgrid = False),
	            zaxis = dict(range = [0,1], title = 'BLUE', showgrid = False),
	            aspectmode='cube'), legend_font_size = 20)
	
	fig = go.Figure(layout = layout)

	if ocs is not None:
		fig.add_trace(mesh_from_hull(ocs, name = 'ocs', title_text = 'Object Colour Solid'))

	if color_predictions is not None:
		for ind, color in enumerate(color_predictions):
			title_text = rgb_to_string(color)
			fig.add_trace(go.Scatter3d(x = [color[0]], 
									   y = [color[1]], 
									   z = [color[2]],
						mode = 'markers', name = 'Predicted color',
			            legendgrouptitle_text = title_text,
			            legendgroup = title_text))
			fig.add_trace(mesh_from_hull(colors_mmb[ind], name = f'Metamer', title_text = title_text))
	else: 
		for ind, color in enumerate(colors_mmb):
			fig.add_trace(mesh_from_hull(color, name = f"Metamer Body{ind + 1}"))
		
	fig.update_traces(showlegend = True)
	fig.show()

	return fig

