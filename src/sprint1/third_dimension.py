import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Cube3DApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Cube Viewer")

        # Create a figure and a 3D axis
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Cube vertices
        self.vertices = np.array([[-1, -1, -1],
                                  [1, -1, -1],
                                  [1, 1, -1],
                                  [-1, 1, -1],
                                  [-1, -1, 1],
                                  [1, -1, 1],
                                  [1, 1, 1],
                                  [-1, 1, 1]])

        # Cube edges
        self.edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
        ]

        # Initial scale and offset for the "infinite" effect
        self.scale = 1.0
        self.offset = np.array([0, 0, 0])


        # Create canvas to embed the figure into Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # Now it's correctly initialized
        self.canvas.get_tk_widget().pack()
        # Plot the cube
        self.plot_cube()

        # Add buttons for interaction
        self.add_buttons()

        # Add zoom functionality
        self.canvas.get_tk_widget().bind("<MouseWheel>", self.zoom)

        # Add pan functionality
        self.canvas.get_tk_widget().bind("<B1-Motion>", self.pan)

    def plot_cube(self):
        # Clear the axis
        self.ax.cla()

        # Adjust the vertices based on scale and offset
        scaled_vertices = self.vertices * self.scale + self.offset

        # Plot each edge of the cube
        for edge in self.edges:
            self.ax.plot3D(*zip(*scaled_vertices[edge]), color="b")

        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)

        # Set the view angle
        self.ax.view_init(30, 30)

        # Redraw the canvas
        self.canvas.draw()

    def add_buttons(self):
        # Create rotation buttons
        self.rotate_x_btn = tk.Button(self.root, text="Rotate X", command=self.rotate_x)
        self.rotate_x_btn.pack(side=tk.LEFT, padx=5)

        self.rotate_y_btn = tk.Button(self.root, text="Rotate Y", command=self.rotate_y)
        self.rotate_y_btn.pack(side=tk.LEFT, padx=5)

        self.rotate_z_btn = tk.Button(self.root, text="Rotate Z", command=self.rotate_z)
        self.rotate_z_btn.pack(side=tk.LEFT, padx=5)

    def rotate_x(self):
        # Rotate cube around X axis
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(np.pi/10), -np.sin(np.pi/10)],
                                    [0, np.sin(np.pi/10), np.cos(np.pi/10)]])
        self.vertices = np.dot(self.vertices, rotation_matrix)
        self.plot_cube()

    def rotate_y(self):
        # Rotate cube around Y axis
        rotation_matrix = np.array([[np.cos(np.pi/10), 0, np.sin(np.pi/10)],
                                    [0, 1, 0],
                                    [-np.sin(np.pi/10), 0, np.cos(np.pi/10)]])
        self.vertices = np.dot(self.vertices, rotation_matrix)
        self.plot_cube()

    def rotate_z(self):
        # Rotate cube around Z axis
        rotation_matrix = np.array([[np.cos(np.pi/10), -np.sin(np.pi/10), 0],
                                    [np.sin(np.pi/10), np.cos(np.pi/10), 0],
                                    [0, 0, 1]])
        self.vertices = np.dot(self.vertices, rotation_matrix)
        self.plot_cube()

    def zoom(self, event):
        # Zoom functionality
        if event.delta > 0:
            self.scale *= 1.1  # Zoom in
        else:
            self.scale /= 1.1  # Zoom out

        self.plot_cube()

    def pan(self, event):
        # Pan functionality (drag to move the view)
        self.offset += np.array([event.x - self.canvas.get_tk_widget().winfo_width() / 2,
                                 event.y - self.canvas.get_tk_widget().winfo_height() / 2,
                                 0]) * 0.01

        self.plot_cube()

# Create the Tkinter window and run the app
root = tk.Tk()
app = Cube3DApp(root)
root.mainloop()
