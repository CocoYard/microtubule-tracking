import PIL.ImageDraw
import napari
from skimage import io
from PIL import Image
import numpy as np
from lib.load_data import TiffLoader
from lib import image_processing
from magicgui import magic_factory
from magicgui import magicgui
import typing


@magicgui(
    call_button="Calculate",
    options={"choices": ['show segmentation', 'show skeleton', 'test']}
)
def widget_demo(
        draw_layer: 'napari.layers.Shapes',
        image_layer: 'napari.layers.Image',
        blur_degree: int,
        blur_method: str,
        options='show segmentation'
) -> typing.List[napari.types.LayerDataTuple]:
    if draw_layer is not None:
        lines = draw_layer.data
        frame2line = {}
        for line in lines:
            frame2line[line[0][0]] = line

    # prepare video data: np.array
    video = np.array(image_layer.data)
    tempv = video.copy()

    tiff_loader = TiffLoader(video)
    # select a line to detect the corresponding microtubule
    for i in range(41, 100):
        img0 = tiff_loader.tiff_gray_image[i]
        if i in frame2line:
            line = frame2line[i]
        if line is None:
            return
        end_points, skltn, thres_img, denoise, temp = image_processing.detectLine(img0, line, blur_degree,
                                                                                  blur_method)  ##[106,347],[112,438]     [44,63],[77,150]    [405,352],[418,440] [287,85],[411,97
        line = [[i, end_points[0][0], end_points[0][1]], [i, end_points[1][0], end_points[1][1]]]
        video[i] = thres_img * 257
        tempv[i] = temp * 257
        print(i)

    layer_type = 'image'
    metadata = {
        'name': 'segment',
        'colormap': 'blue',
        'opacity': 0.4
    }
    metadata1 = {
        'name': 'segment2',
        'colormap': 'gray'
    }
    return [(video, metadata, layer_type), (tempv, metadata1, layer_type)]


@magicgui(call_button="Save to local", )
def save(image_layer: 'napari.layers.Image', ):
    image_layer.save('segmentation.tif', plugin='builtins')


viewer = napari.Viewer()
viewer.window.add_dock_widget(widget_demo)
viewer.window.add_dock_widget(save)
napari.run()
