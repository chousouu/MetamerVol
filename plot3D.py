import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import MetamerMismatch as mm

def plot_metamer_body(mmbs, ocs):
	def mesh_from_hull(hull, name = None): #color = None works, it sets random colors

		vertices = hull.points 
		faces = hull.simplices
		
		color = None if name != 'ocs' else 'black'

		mesh = \
			go.Mesh3d(
			x=np.array(vertices)[:,0],
			y=np.array(vertices)[:,1],
			z=np.array(vertices)[:,2],
			i=np.array(faces)[:,0],
			j=np.array(faces)[:,1],
			k=np.array(faces)[:,2],
			opacity=0.5,
			color = color, name = name) #,  alphahull = 0 # ?
		
		return mesh

	layout = \
		go.Layout(
	        scene = dict(
	            xaxis = dict(range = [0,1], title = 'r', showgrid = False),
	            yaxis = dict(range = [0,1], title = 'g', showgrid = False),
	            zaxis = dict(range = [0,1], title = 'b', showgrid = False),
	            aspectmode='cube'), legend_font_size = 20)

	meshes = []

	meshes.append(mesh_from_hull(ocs, 'ocs'))

	for mmb in mmbs:
		meshes.append(mesh_from_hull(mmb))

	fig = go.Figure(data = meshes, layout = layout)
	fig.update_traces(showlegend=True, showscale=False)


	return fig

def plot_scene(ocs, colors_mmb, color_prediction = None):
	fig = plot_metamer_body(mmbs = colors_mmb, ocs = ocs)

	if color_prediction is not None:
		fig.add_trace(go.Scatter3d(x = [color_prediction[0]], y = [color_prediction[1]], z = [color_prediction[2]],
			     mode = 'markers', name = 'predicted color'))
		
	fig.show()

	return

#maybe i want to use it?..
def rgb_to_string(rgb):
    return '_'.join([(str(format(_, '.3f'))) for _ in rgb])
