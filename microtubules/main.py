import PIL.ImageDraw
import napari
import numpy as np
from lib.load_data import TiffLoader
from lib import image_processing
from magicgui import magicgui
from lib.img_helper import darken
import matplotlib.pyplot as plt
import typing
from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari.utils.notifications import (
    ErrorNotification,
    Notification,
    NotificationSeverity,
    notification_manager,
)

@magicgui(
    call_button="Calculate",
)
def main(
        draw_layer: 'napari.layers.Shapes',
        image_layer: 'napari.layers.Image',
        struct_size=7,
        start_frame=0,
        end_frame=121,
        thres_ratio=1.0
) -> typing.List[napari.types.LayerDataTuple]:
    frame2line = {}
    if draw_layer is not None:
        lines = draw_layer.data
        for line in lines:
            frame2line[line[0][0]] = line
    # prepare video data: np.array
    video = np.array(image_layer.data)
    length = []

    tiff_loader = TiffLoader(video)
    if start_frame not in frame2line:
        notif = Notification(
            'Please draw a line to specify a microtubule at your start frame',
            NotificationSeverity.WARNING,
            actions=[('Got it', lambda x: None)],
        )
        NapariQtNotification.show_notification(notif)
        return
    # select a line to detect the corresponding microtubule
    for i in range(start_frame, min(end_frame, len(video))):
        img = tiff_loader.tiff_gray_image[i]
        if i in frame2line:
            line = frame2line[i]
        ret = image_processing.detectLine(img, line, struct_size, thres_ratio)
        if ret is None:
            notif = Notification(
                'No detected microtubules at frame ' + str(i) + '. Please draw a line at more frames.',
                NotificationSeverity.WARNING,
                actions=[('OK', lambda x: None)],
            )
            NapariQtNotification.show_notification(notif)
            return
        end_points, thres_img, l = ret
        line = [[i, end_points[0][0], end_points[0][1]], [i, end_points[1][0], end_points[1][1]]]
        video[i] = thres_img * 257
        length.append(l)
        print(i)
    length = np.array(length)
    x = range(len(length))
    plt.plot(x, length, color='r')
    plt.legend(loc="best")
    plt.show()

    layer_type = 'image'
    metadata3 = {
        'name': 'segment',
        'colormap': 'red',
        'blending': 'additive'
    }
    return [(video, metadata3, layer_type)]


@magicgui(
    call_button="Darken"
)
def darken_img(
        dark_layer: 'napari.layers.Shapes',
        image_layer: 'napari.layers.Image',
        dark_ratio=84,
        start_frame=0,
        end_frame=68
) -> napari.types.LayerDataTuple:
    # the whole output img may look not like the original one. It is the Napari's displaying issue but
    # the pixel values are the same.

    # prepare video data: np.array
    video = np.array(image_layer.data)
    tiff_loader = TiffLoader(video)
    polygon = dark_layer.data[0]
    polygon = polygon[:, 1:]
    for i in range(start_frame, min(end_frame, len(video))):
        img = tiff_loader.tiff_gray_image[i]
        darken(polygon, img, dark_ratio)
        video[i] = img * 257
        print(i)
    layer_type = 'image'
    metadata = {
        'name': 'darken',
        'colormap': 'gray'
    }

    return video, metadata, layer_type


@magicgui(call_button="Save to local", )
def save(image_layer: 'napari.layers.Image', ):
    image_layer.save('segmentation.tif', plugin='builtins')


viewer = napari.Viewer()
viewer.window.add_dock_widget(main)
viewer.window.add_dock_widget(darken_img)
viewer.window.add_dock_widget(save)
napari.run()
