import plotly.offline as py
import plotly.graph_objs as go
import numpy as np

def mesh_from_hull(hull, color, name):
	vertices = hull.points 
	faces = hull.simplices
	mesh = \
		go.Mesh3d(
		   x=np.array(vertices)[:,0],
		   y=np.array(vertices)[:,1],
		   z=np.array(vertices)[:,2],
		   i=np.array(faces)[:,0],
		   j=np.array(faces)[:,1],
		   k=np.array(faces)[:,2],
		   opacity=0.5,
		   color=color, name=name) #,  alphahull = 0
	return mesh

def plot_convex_hulls(hulls, filename, showOCS = False):
	layout = \
		go.Layout(
	        scene = dict(
	            xaxis = dict(range = [0,1], title = 'r', showgrid = False),
	            yaxis = dict(range = [0,1], title = 'g', showgrid = False),
	            zaxis = dict(range = [0,1], title = 'b', showgrid = False),
	            aspectmode='cube'))

	meshes = [mesh_from_hull(h, color, name) for h, color, name in hulls]

	fig = go.Figure(data=meshes, layout=layout)
	fig.update_traces(showlegend=True, showscale=False)
	fig.update_layout(legend_font_size = 20)

	fig.show()
	# py.plot(fig, filename=filename)
