import PIL.ImageDraw
import napari
from skimage import io
from PIL import Image
import numpy as np
from lib.load_data import TiffLoader
from lib import image_processing
from magicgui import magic_factory
from magicgui import magicgui


@magicgui(
    call_button="Calculate",
    options={"choices": ['show segmentation', 'show skeleton']}
)
def widget_demo(
    draw_layer: 'napari.layers.Shapes',
    image_layer: 'napari.layers.Image',
    options='show segmentation'
) -> 'napari.types.LayerDataTuple':
    # prepare lines data: list(np.array)
    lines = draw_layer.data
    lines.sort(key=lambda x: x[0][0])
    # prepare video data: np.array
    video = np.array(image_layer.data)

    tiff_loader = TiffLoader(video)
    img0 = tiff_loader.tiff_gray_image[0]
    # select a line to detect the corresponding microtubule
    line = lines[0]
    img, end_points, skltn, thres_img = image_processing.detectLine(img0, line)  ##[106,347],[112,438]     [44,63],[77,150]    [405,352],[418,440] [287,85],[411,97

    img0 = Image.fromarray(img0.astype(np.uint8))
    draw = PIL.ImageDraw.Draw(img0)
    draw.point(end_points, fill="RED")
    img.show()

    layer_type = 'image'
    metadata = {
        'name': 'segment',
        'colormap': 'gray'
    }

    if options == 'show segmentation':
        video[0] = thres_img
    else:
        video[0] = skltn
        metadata['name'] = 'skeleton'
    for i in range(len(video[0])):
        for j in range(len(video[0][i])):
            if video[0][i][j] != 0:
                video[0][i][j] = 65535

    return video, metadata, layer_type


@magicgui(call_button="Save to local",)
def save(image_layer: 'napari.layers.Image',):
    image_layer.save('segmentation.tif', plugin='builtins')


viewer = napari.Viewer()
viewer.window.add_dock_widget(widget_demo)
viewer.window.add_dock_widget(save)
napari.run()
