from Utils import *
from pprint import pp


PATH = 'scenes/resources/small_scene.glb'


# TODO
# properly setup scene based on gltf data
# support transparency


import io
import base64
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

def get_image_bytes(gltf, image, base_dir = None):
	if image.uri:
		if image.uri.startswith("data:"):
			return base64.b64decode(image.uri.split(",", 1)[1])
		return (Path(base_dir) / image.uri).read_bytes()
		
	if image.bufferView is not None:
		bv = gltf.bufferViews[image.bufferView]
		buffer = gltf.buffers[bv.buffer]
		
		# Embedded binary chunk in GLB
		if buffer.uri is None:
			blob = gltf.binary_blob()
			return blob[bv.byteOffset : bv.byteOffset + bv.byteLength]
		
		# External .bin file
		bin_path = Path(base_dir) / buffer.uri
		with open(bin_path, "rb") as f:
			f.seek(bv.byteOffset)
			return f.read(bv.byteLength)
			
	return None

def display_textures(gltfm, base_dir = None):
	
	root = tk.Tk()
	root.title("glTF Textures Viewer")
	
	canvas = tk.Canvas(root)
	scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
	frame = tk.Frame(canvas)
	
	frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
	canvas.create_window((0, 0), window=frame, anchor="nw")
	canvas.configure(yscrollcommand=scrollbar.set)
	
	canvas.pack(side="left", fill="both", expand=True)
	scrollbar.pack(side="right", fill="y")
	
	# Reference list to prevent Tkinter garbage collection of PhotoImages
	tk_images = []
	
	for idx, image_def in enumerate(gltf.images):
		img_bytes = get_image_bytes(gltf, image_def, base_dir)
		if not img_bytes:
			continue
			
		pil_img = Image.open(io.BytesIO(img_bytes))
		pil_img.thumbnail((256, 256))
		
		tk_img = ImageTk.PhotoImage(pil_img)
		tk_images.append(tk_img)
		
		lbl = tk.Label(frame, image=tk_img, text=f"Image {idx}", compound="bottom")
		lbl.grid(row=idx // 3, column=idx % 3, padx=5, pady=5)
		
	root.mainloop()

gltf = GLTF2().load(PATH)
base_dir = Path(PATH).parent
print("node name, node mesh id")
pp([ (node.name, node.mesh) for node in gltf.nodes ])
print("mesh name, prinitive count")
pp([ (mesh.name, len(mesh.primitives)) for mesh in gltf.meshes ])
#print("material id, material data")
#pp([ (prim.material, gltf.materials[prim.material].to_dict()) for mesh in gltf.meshes for prim in mesh.primitives ])
#pp([ mat.to_dict() for mat in enumerate(gltf.materials) ])

display_textures(gltf) #, base_dir)