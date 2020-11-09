from imesotp import OTPClient
import numpy as np

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtCore import QTimer
import sys
from pipython import GCSDevice, pitools
import time
import os


class Hexapod:
    def __init__(self, ip_address, verbose=False):
        self._ip_address = ip_address
        self._contoller_name = 'C-887'  # can leave empty
        self._pidevice = GCSDevice(self._contoller_name)
        self._ranges = []
        self._current_pos = []
        self._current_vel = []
        self._current_pivot = []
        self._verbose = verbose
        self._is_init = False

    def __enter__(self):
        self._init_hexapod()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_hexapod()

    def _init_hexapod(self):
        self._setup_hexapod()
        self._fast_reference_move()
        self._is_init = True

    def _close_hexapod(self):
        self._is_init = False
        self._pidevice.CloseConnection()

    def _setup_hexapod(self):
        if self._verbose:
            print("Connecting to Hexapod...")

        self._pidevice.ConnectTCPIP(ipaddress=self._ip_address)

        if self._verbose:
            print('connected: {}'.format(self._pidevice.qIDN().strip()))
            if self._pidevice.HasqVER():
                print('version info: {}'.format(self._pidevice.qVER().strip()))

    def _fast_reference_move(self):
        if self._verbose:
            print('initialize connected stages...')

        self._pidevice.SVO(self._pidevice.axes, [True] * len(self._pidevice.axes))
        self._pidevice.FRF()  # start a reference move

        if self._verbose:
            print('waiting for calibration...')

        pitools.waitonready(self._pidevice)
        rangemin = list(self._pidevice.qTMN(self._pidevice.axes).values())
        rangemax = list(self._pidevice.qTMX(self._pidevice.axes).values())
        self._ranges = list(zip(rangemin, rangemax))

        if self._verbose:
            print('ranges', list(self._ranges))

    def _assert_is_init(self):
        assert self._is_init, "Hexapod is not initialized"

    def set_target(self, targets, wait_on_target=True):
        """

        :param targets: list with targets [x, y, z, u, v, w]
        :param wait_on_target: bool, if True, this method blocks until targets are reached
        :return:
        """
        self._assert_is_init()

        if self._verbose:
            print('move stages...')

        self._pidevice.MOV(self._pidevice.axes, targets)

        if wait_on_target:
            self.wait_on_target()

        self._current_pos = self._pidevice.qPOS()
        if self._verbose:
            for axis in self._current_pos:
                print('position of axis {} = {}'.format(axis, self._current_pos[axis]))

        return self._current_pos

    def set_velocity(self, velocity: float):
        self._assert_is_init()
        if self._verbose:
            print('setting velocity to', velocity)
        self._pidevice.VLS(velocity)
        self.get_velocity()

    def get_velocity(self):
        self._assert_is_init()
        self._current_vel = self._pidevice.qVLS()
        if self._verbose:
            print('current velocity is', self._current_vel)
        return self._current_vel

    def set_pivot(self, pivot):
        self._assert_is_init()
        assert len(pivot) == 3, "pivot size mismatch"
        self._pidevice.SPI(['R', 'S', 'T'], pivot)
        self.get_pivot()

    def get_pivot(self):
        self._current_pivot = self._pidevice.qSPI()
        return self._current_pivot

    def wait_on_target(self):
        self._assert_is_init()
        if self._verbose:
            print('Waiting to be on target...')
        pitools.waitontarget(self._pidevice)

    def get_current_positions(self):
        self._assert_is_init()
        self._current_pos = self._pidevice.qPOS()
        if self._verbose:
            for axis in self._current_pos:
                print('position of axis {} = {}'.format(axis, self._current_pos[axis]))
        return self._current_pos

    def is_on_target(self):
        self._assert_is_init()
        return self._pidevice.qONT(self._pidevice.axes)


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


def control():
    num_imgs = 1

    with OTPClient('192.168.127.1', 5005, verbose=False) as oct_client:
        print("Calibrating Hexapod, please wait...")
        with Hexapod('169.254.7.95') as hexapod:

            def make_getter():
                hexapod.set_pivot([0.0, 0.0, 35.0])
                base_targets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                targets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                hexapod.set_target(base_targets)
                img = np.array([])
                counter = 0

                def img_getter():
                    nonlocal img, counter

                    # generate some random target
                    targets[0] = base_targets[1] + np.random.rand() * 0.0  # X
                    targets[1] = base_targets[1] + np.random.rand() * 0.0  # Y
                    targets[2] = base_targets[1] + np.random.rand() * 0.0  # Z
                    targets[3] = base_targets[3] + np.random.rand() * 0.0  # U=0
                    targets[4] = base_targets[4] + np.random.rand() * 0.0  # V=0
                    targets[5] = base_targets[5] + np.random.rand() * 0.0  # W=0

                    # set hexapod to target
                    hexapod.set_target(targets)

                    # get OCT image
                    img = oct_client.get_image()

                    # get current position from hexapod
                    position = list(hexapod.get_current_positions().values())

                    # save OCT image and position to file
                    np.savez_compressed(os.path.join('/home/mohanxu/Desktop/ophonlas-oct/oct_network_app/original/test20', '1'), data=img, pos=position)

                    # increment counter
                    counter = counter + 1
                    print(counter, position)
                    if counter == num_imgs:
                        exit()

                    return img

                return img_getter

            qt_imshow(make_getter(), refresh_rate=33)

def control_valid(x, y, z):
    num_imgs = 1

    with OTPClient('192.168.127.1', 5005, verbose=False) as oct_client:
        print("Calibrating Hexapod, please wait...")
        with Hexapod('169.254.7.95') as hexapod:

            def make_getter():
                hexapod.set_pivot([0.0, 0.0, 35.0])
                base_targets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                targets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                hexapod.set_target(base_targets)
                img = np.array([])
                counter = 0

                def img_getter():
                    nonlocal img, counter

                    # generate some random target
                    targets[0] = x  # X
                    targets[1] = y  # Y
                    targets[2] = z  # Z
                    targets[3] = base_targets[3] + np.random.rand() * 0.0  # U=0
                    targets[4] = base_targets[4] + np.random.rand() * 0.0  # V=0
                    targets[5] = base_targets[5] + np.random.rand() * 0.0  # W=0

                    # set hexapod to target
                    hexapod.set_target(targets)

                    # get OCT image
                    img = oct_client.get_image()

                    # get current position from hexapod
                    position = list(hexapod.get_current_positions().values())

                    # save OCT image and position to file
                    np.savez_compressed(os.path.join('/home/mohanxu/Desktop/code_mohan2/Data_VS/Test', '1'), data=img, pos=position)

                    # increment counter
                    counter = counter + 1
                    #print(counter, position)
                    if counter == num_imgs:
                        time.sleep(1)

                    return img

                return img_getter

            qt_imshow(make_getter(), refresh_rate=33)

if __name__ == '__main__':
    control()
