#!/usr/bin/python

from __future__ import print_function

import time
import usb.core
import usb.util

try:
    user_input = raw_input
except NameError:
    user_input = input

# Unknown
# vendor id = 0x46d, product id=0xc215

# Yizish
# Doesn't currently work

# Onyx (5lb)
# vendor id = 0x1446, product id=0x6a73

# Pronto Digital (large light grey)
# Doesn't currently work

# I-2000 (smallest light grey)
# Doesn't currently work

# https://gist.github.com/jacksenechal/5862530

#ONYX_VENDOR_ID = 0x1446 # hex
#ONYX_PRODUCT_ID = 0x6a73 # hex
ONYX_VENDOR_ID = 5190 # hex
ONYX_PRODUCT_ID = 27251 # hex

INTERFACE = 0
DATA_MODE_GRAMS = 2
DATA_MODE_OUNCES = 11

#IGNORE_DEVICES = [
#    (32903, 32770),
#    (7531, 2),
#    (7531, 3),
#    (1133, 50475),
#    (32903, 32778),
#    (1133, 49685),
#]

# https://github.com/pyusb/pyusb/blob/master/docs/tutorial.rst

# TODO: buses are the same but the addresses change
ARIADNE_BACK_USB = 1
ARIADNE_FRONT_USB = 3

# https://github.com/pyusb/pyusb/blob/master/docs/tutorial.rst#dealing-with-multiple-identical-devices
# http://steventsnyder.c,om/reading-a-dymo-usb-scale-using-python/
# https://pypi.org/project/python-dymo-scale/

# make sure you add the right permission to /etc/udev/rules.d/
class Scale(object):
    def __init__(self, device):
        self.dev = device
        #self.dev = usb.core.find(idVendor=vendor_id, idProduct=product_id)
        if self.dev is None:
            raise ValueError('Scale is not connected.')
        self.endpoint = self.dev[0][(0, 0)][0]

    def __enter__(self):
        if self.dev.is_kernel_driver_active(INTERFACE) is True:
            self.dev.detach_kernel_driver(INTERFACE)
            self.dev.set_configuration()
            usb.util.claim_interface(self.dev, INTERFACE)

    def __exit__(self, exc_type, exc_val, exc_tb):
        usb.util.release_interface(self.dev, INTERFACE)
        self.dev.attach_kernel_driver(INTERFACE)

    def read(self): #, attempts=10):
        raw_data = None
        while raw_data is None:
            raw_data = self.endpoint.read(self.endpoint.wMaxPacketSize)
            #print(raw_data)
            #attempts -= 1
            if not raw_data:
                continue
            mode = raw_data[2]
            raw_weight = raw_data[4] + raw_data[5] * 256
            raw_weight = -raw_weight if raw_data[1] == 5 else raw_weight
            if mode == DATA_MODE_OUNCES:
                weight = raw_weight * 0.1
            elif mode == DATA_MODE_GRAMS:
                weight = raw_weight
            else:
                raise ValueError(mode)
            return weight

    def stable_read(self, min_stable_count=2):
        not_stable = True
        prev = None
        stable_count = 0
        while not_stable or (stable_count < min_stable_count):
            curr = self.read()
            if prev == curr:
                stable_count += 1
                not_stable = False
            else:
                stable_count = 0
                not_stable = True
            prev = curr
        return curr

def find_scales(vendor=None, product=None, bus=None):
    #print(usb.core.show_devices())
    for device in sorted(usb.core.find(find_all=True), key=lambda d: d.bus):
        #print(device)
        #if (device.idVendor, device.idProduct) in IGNORE_DEVICES:
        #    continue
        #print(device)
        if (vendor is not None) and (device.idVendor != vendor):
            continue
        if (product is not None) and (device.idProduct != product):
            continue
        if (bus is not None) and (device.bus != bus):
            continue
        #print(device.bus, device.address)
        #print('vendor id = {}, product id={}'.format(hex(device.idVendor), hex(device.idProduct)))
        print('vendor id = {}, product id={}, bus={}'.format(device.idVendor, device.idProduct, device.bus))
        yield device

def read_scales():
    weights = {}
    for device in find_scales(ONYX_VENDOR_ID, ONYX_PRODUCT_ID):
        try:
            scale = Scale(device)
            with scale:
                weights[device.bus] = scale.stable_read()
                #print('Bus: {}, weight={}oz'.format(device.bus, weights[device.bus]))
        except usb.core.USBError as e:
            print(e)
    return weights

def main():
    read_scales()

if __name__ == '__main__':
    main()
    #find_all_scales()