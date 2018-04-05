import os
import numpy as np
from pynq import Overlay, PL, MMIO
from pynq import DefaultIP, DefaultHierarchy
from pynq import Xlnk
from pynq.lib import DMA

CV2PYNQ_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
CV2PYNQ_BIT_DIR = os.path.join(CV2PYNQ_ROOT_DIR, 'bitstreams')

class cv2pynq():
    MAX_WIDTH  = 1920
    MAX_HEIGHT = 1080
    def __init__(self, load_overlay=True):
        self.bitstream_name = None
        self.bitstream_name = "image_filters_01.bit"
        self.bitstream_path = os.path.join(CV2PYNQ_BIT_DIR, self.bitstream_name)
        #if PL.bitfile_name != self.bitstream_path:#todo ol
        #if load_overlay:
        self.ol = Overlay(self.bitstream_path)
        self.ol.download()
        self.ol.reset() # todo
        print("downloaded overlay", self.bitstream_path)
        #else:
        #    raise RuntimeError("Incorrect Overlay loaded")
        self.xlnk = Xlnk()
        self.input_buffer  = self.xlnk.cma_array(shape=(self.MAX_HEIGHT,self.MAX_WIDTH), dtype=np.uint8)
        self.output_buffer = self.xlnk.cma_array(shape=(self.MAX_HEIGHT,self.MAX_WIDTH), dtype=np.uint8)
        self.dmaOut = self.ol.image_filters.axi_dma_0.sendchannel 
        self.dmaIn =  self.ol.image_filters.axi_dma_0.recvchannel 
        self.dmaOut.stop()#todo
        self.dmaIn.stop()
        self.dmaIn.start()
        self.dmaOut.start()
        self.filter = -1 # filter types: SobelX = 1, SobelY = 2

    #def __del__(self):
    #    self.input_buffer.close()
    #    self.output_buffer.close()
    #    self.dmaOut.stop()
    #    self.dmaIn.stop()
    #    print("_del_")
    
    def filter2D(self, src):
        f2D = self.ol.image_filters.filter2D_hls_0
        f2D.start()
        np.copyto(self.output_buffer,src)
        self.dmaIn.transfer(self.input_buffer)
        self.dmaOut.transfer(self.output_buffer)
        self.dmaIn.wait()
        return self.input_buffer.copy()

    def Sobel(self,src, ddepth, dx, dy):
        #print(self.bitstream_name)
        #print(type(self.ol))
        f2D = self.ol.image_filters.filter2D_hls_0
        #print(src.shape[0],src.shape[1])
        f2D.rows = src.shape[0]
        f2D.columns = src.shape[1]
        f2D.channels = 1
        if (self.filter != 1) and (dx == 1) and (dy == 0) :
            self.filter = 1
            f2D.r1 = 0x000100ff #[-1  0  1]
            f2D.r2 = 0x000200fe #[-2  0  2]
            f2D.r3 = 0x000100ff #[-1  0  1]
        elif (self.filter != 2) and (dx == 0) and (dy == 1) :
            self.filter = 2
            f2D.r1 = 0x00fffeff #[-1 -2 -1]
            f2D.r2 = 0x00000000 #[ 0  0  0]
            f2D.r3 = 0x00010201 #[ 1  2  1]
        else:
            raise RuntimeError("Incorrect dx dy configuration")
        
        return self.filter2D(src)

class cv2pynqDiverImageFilters(DefaultHierarchy):
    def __init__(self, description):
        super().__init__(description)

    def image_filter(self, rows, columns):
        self.filter2D_hls_0.rows = rows
        self.filter2D_hls_0.columns = columns
        
        self.axi_dma_0.sendchannel.transfer(columns)
        #self.axi_dma_0.recvchannel.transfer(out_buffer)
        #self.multiply_dma.sendchannel.wait()
        #self.multiply_dma.recvchannel.wait()
        return "worksWell"

    @staticmethod
    def checkhierarchy(description):
        if 'axi_dma_0' in description['ip'] \
           and 'filter2D_hls_0' in description['ip']:
            return True
        return False

class cv2pynqDriverFilter2D(DefaultIP):
    def __init__(self, description):
        super().__init__(description=description)
        
    bindto = ['xilinx.com:hls:filter2D_hls:1.0']

    def start(self):
        self.write(0x00, 0x01)#todo 0x81 restart

    @property
    def rows(self):
        return self.read(0x14)
    @rows.setter
    def rows(self, value):
        self.write(0x14, value)

    @property
    def columns(self):
        return self.read(0x1c)
    @columns.setter
    def columns(self, value):
        self.write(0x1c, value)

    @property
    def channels(self):
        return self.read(0x24)
    @channels.setter
    def channels(self, value):
        self.write(0x24, value)
    
    @property
    def mode(self):
        return self.read(0x2c)
    @mode.setter
    def mode(self, value):
        self.write(0x2c, value)
    
    @property
    def r1(self):
        return self.read(0x34)
    @r1.setter
    def r1(self, value):
        self.write(0x34, value)

    @property
    def r2(self):
        return self.read(0x3c)
    @r2.setter
    def r2(self, value):
        self.write(0x3c, value)

    @property
    def r3(self):
        return self.read(0x44)
    @r3.setter
    def r3(self, value):
        self.write(0x44, value)
    