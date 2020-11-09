from imesotp import OTPClient
import numpy as np

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtCore import QTimer
import sys


def qt_imshow(img_fun, refresh_rate=100):

    app = QApplication(sys.argv)
    label = QLabel()
    label.show()
    label.resize(512, 512)

    def update_img():
        try:
            img = img_fun()
            height, width = img.shape
            img = np.array(img, dtype=np.uint8, copy=False, order='C')
            channel = 1
            bytes_per_line = channel * width
            qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_Indexed8)
            pixmap = QPixmap(qimg).scaled(height, height)
            label.resize(pixmap.width(), pixmap.height())
            label.setPixmap(pixmap)
        finally:
            QTimer.singleShot(refresh_rate, update_img)

    update_img()

    app.exec_()  # drops into event loop


with OTPClient('192.168.127.1', 5005, verbose=False) as client:

    def img_getter():
        client.set_pos(np.random.rand() * 0.0,
                       np.random.rand() * 0.0,
                       np.random.rand() * 0.0)  # move OCT or hexapod here
        img = client.get_image()
        assert len(img.shape) == 2
        return img

    qt_imshow(img_getter, refresh_rate=33)

