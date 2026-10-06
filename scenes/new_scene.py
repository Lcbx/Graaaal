#from Utils import *

from pprint import pp
import io
import base64
from pathlib import Path
from pygltflib import GLTF2, BufferView, Accessor
from PIL import Image, ImageTk


PATH = 'scenes/resources/small_scene.glb'


# TODO
# properly setup scene based on gltf data
# support transparency


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

def uses_alpha_transparency(gltf, mat, base_dir):
	if mat.alphaMode in ('MASK', 'BLEND'):
		return True
	if mat.pbrMetallicRoughness:
		pbr = mat.pbrMetallicRoughness
		if pbr.baseColorFactor and pbr.baseColorFactor[3] < 1.0:
			return True
		#print("hit", mat.name)
		if pbr.baseColorTexture is not None:
			img_obj = gltf.images[gltf.textures[pbr.baseColorTexture.index].source]
			img_data = get_image_bytes(gltf, img_obj, base_dir)
			if img_data and Image.open(io.BytesIO(img_data)).convert('RGBA').getchannel('A').getextrema()[0] < 255:
				return True
	return False

gltf = GLTF2().load(PATH)
base_dir = Path(PATH).parent
print("node name, node mesh id")
pp([ (node.name, node.mesh) for node in gltf.nodes ])
print("mesh name, primitive count")
pp([ (mesh.name, len(mesh.primitives)) for mesh in gltf.meshes ])
print("node name, mesh id, primitive id, material id")
pp([ (node.name, node.mesh, prim_id, prim.material) for node in gltf.nodes if node.mesh is not None for prim_id, prim in enumerate(gltf.meshes[node.mesh].primitives) ])
#pp([ mat.to_dict() for mat in enumerate(gltf.materials) ])
print("material id, name, uses alpha")
pp([(i, mat.name, uses_alpha_transparency(gltf, mat, base_dir)) for i, mat in enumerate(gltf.materials)])


def display_textures(gltf, base_dir = None):
	import tkinter as tk
	
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

#display_textures(gltf) #, base_dir)