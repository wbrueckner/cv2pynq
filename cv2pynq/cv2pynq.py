import os
import numpy as np
from pynq import Overlay, PL, MMIO
from pynq import DefaultIP, DefaultHierarchy
from pynq import Xlnk
from pynq.lib import DMA
from cffi import FFI
import cv2

CV2PYNQ_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
CV2PYNQ_BIT_DIR = os.path.join(CV2PYNQ_ROOT_DIR, 'bitstreams')

class cv2pynq():
    MAX_WIDTH  = 1920
    MAX_HEIGHT = 1080
    def __init__(self, load_overlay=True):
        self.bitstream_name = None
        self.bitstream_name = "filter2D_modes12.bit"
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
        #self.input_buffer  = self.xlnk.cma_array(shape=(self.MAX_HEIGHT,self.MAX_WIDTH), dtype=np.uint8)
        self.partitions = 10 #split the cma into partitions for pipelined transfer
        self.cmaPartitionLen = self.MAX_HEIGHT*self.MAX_WIDTH/self.partitions
        self.listOfcma = [self.xlnk.cma_array(shape=(int(self.MAX_HEIGHT/self.partitions),self.MAX_WIDTH), dtype=np.uint8) for i in range(self.partitions)]
        self.output_buffer = self.xlnk.cma_array(shape=(self.MAX_HEIGHT,self.MAX_WIDTH), dtype=np.uint8)
        self.dmaOut = self.ol.image_filters.axi_dma_0.sendchannel 
        self.dmaIn =  self.ol.image_filters.axi_dma_0.recvchannel 
        self.dmaOut.stop()#todo
        self.dmaIn.stop()
        self.dmaIn.start()
        self.dmaOut.start()
        self.filter = "" # filter types: SobelX, SobelY, ScharrX, ScharrY
        self.ffi = FFI()
        self.ol.image_filters.filter2D_kernel_f_0.reset()

    #def __del__(self):
    #    self.input_buffer.close()
    #    self.output_buffer.close()
    #    self.dmaOut.stop()
    #    self.dmaIn.stop()
    #    print("_del_")

    def filter2D(self, src):
        f2D = self.ol.image_filters.filter2D_kernel_f_0
        f2D.start()
        ret = self.xlnk.cma_array(src.shape, dtype=np.uint8)
        if hasattr(src, 'physical_address') :
            self.dmaIn.transfer(ret)
            self.dmaOut.transfer(src)
            self.dmaIn.wait()
        else:#pipeline the copy to continuous memory and filter calculation in hardware
            chunks = int(src.nbytes / (self.cmaPartitionLen) )
            pointerCma = self.ffi.cast("uint8_t *",  self.ffi.from_buffer(self.listOfcma[0]))
            pointerToImage = self.ffi.cast("uint8_t *", self.ffi.from_buffer(src))
            if chunks > 0:
                self.ffi.memmove(pointerCma, pointerToImage, self.listOfcma[0].nbytes)
                self.dmaIn.transfer(ret)
            for i in range(1,chunks):
                while not self.dmaOut.idle and not self.dmaOut._first_transfer:
                    pass 
                self.dmaOut.transfer(self.listOfcma[i-1])
                pointerCma = self.ffi.cast("uint8_t *",  self.ffi.from_buffer(self.listOfcma[i]))
                self.ffi.memmove(pointerCma, pointerToImage+i*self.listOfcma[i].nbytes, self.listOfcma[i].nbytes)
            if chunks > 0:
                while not self.dmaOut.idle and not self.dmaOut._first_transfer:
                    pass 
                self.dmaOut.transfer(self.listOfcma[chunks-1])
            if(src.nbytes % self.cmaPartitionLen != 0):#cleanup code - handle rest of image
                rest = self.xlnk.cma_array(shape=(int(src.nbytes-chunks*self.cmaPartitionLen),1), dtype=np.uint8)
                pointerCma = self.ffi.cast("uint8_t *",  self.ffi.from_buffer(rest))
                self.ffi.memmove(pointerCma, pointerToImage+int(chunks*self.cmaPartitionLen), rest.nbytes)
                while not self.dmaOut.idle and not self.dmaOut._first_transfer:
                    pass 
                self.dmaOut.transfer(rest)
            self.dmaIn.wait()
        return ret

    def Sobel(self,src, ddepth, dx, dy):
        #print(self.bitstream_name)
        #print(type(self.ol))
        f2D = self.ol.image_filters.filter2D_kernel_f_0
        #print(src.shape[0],src.shape[1])
        f2D.rows = src.shape[0]
        f2D.columns = src.shape[1]
        f2D.channels = 1
        if (self.filter != "SobelX") and (dx == 1) and (dy == 0) :
            self.filter = "SobelX"
            f2D.mode = 1
            f2D.r1 = 0x000100ff #[-1  0  1]
            f2D.r2 = 0x000200fe #[-2  0  2]
            f2D.r3 = 0x000100ff #[-1  0  1]
        elif (self.filter != "SobelY") and (dx == 0) and (dy == 1) :
            self.filter = "SobelY"
            f2D.mode = 1
            f2D.r1 = 0x00fffeff #[-1 -2 -1]
            f2D.r2 = 0x00000000 #[ 0  0  0]
            f2D.r3 = 0x00010201 #[ 1  2  1]
        else:
            raise RuntimeError("Incorrect dx dy configuration")  
        return self.filter2D(src)

    def Scharr(self,src, ddepth, dx, dy):
        f2D = self.ol.image_filters.filter2D_kernel_f_0
        f2D.rows = src.shape[0]
        f2D.columns = src.shape[1]
        f2D.channels = 1
        if (self.filter != "ScharrX") and (dx == 1) and (dy == 0) :
            self.filter = "ScharrX"
            f2D.mode = 1
            f2D.r1 = 0x000300fd #[-3  0  3]
            f2D.r2 = 0x000a00f6 #[-10 0 10]
            f2D.r3 = 0x000300fd #[-3  0  3]
        elif (self.filter != "ScharrY") and (dx == 0) and (dy == 1) :
            self.filter = "ScharrY"
            f2D.mode = 1
            f2D.r1 = 0x00fdf6fd #[-3 -10 -3]
            f2D.r2 = 0x00000000 #[ 0   0  0]
            f2D.r3 = 0x00030a03 #[ 3  10  3]
        else:
            raise RuntimeError("Incorrect dx dy configuration")  
        return self.filter2D(src)

    def Laplacian(self,src, ddepth):
        f2D = self.ol.image_filters.filter2D_kernel_f_0
        f2D.rows = src.shape[0]
        f2D.columns = src.shape[1]
        f2D.channels = 1
        if (self.filter != "Laplacian")  :
            self.filter = "Laplacian"
            f2D.mode = 1
            f2D.r1 = 0x00000100 #[ 0  1  0]
            f2D.r2 = 0x0001fc01 #[ 1 -4  1]
            f2D.r3 = 0x00000100 #[ 0  1  0] 
        return self.filter2D(src)
    
    def floatToFixed(self, f, total_bits, fract_bits):
        """convert float f to a signed fixed point with #total_bits and #frac_bits after the point"""
        fix = int((abs(f) * (1 << fract_bits)))
        if(f < 0):
            fix += 1 << total_bits-1
        return fix

    def blur(self,src, ksize):
        f2D = self.ol.image_filters.filter2D_kernel_f_0
        f2D.rows = src.shape[0]
        f2D.columns = src.shape[1]
        if (self.filter != "blur")  :
            self.filter = "blur"
            f2D.mode = 2
            mean = self.floatToFixed(1/9, 16, 14)
            f2D.r11 = mean + (mean << 16)
            f2D.r12 = mean
            f2D.r21 = mean + (mean << 16)
            f2D.r22 = mean
            f2D.r31 = mean + (mean << 16)
            f2D.r32 = mean
        return self.filter2D(src)
    
    def GaussianBlur(self, src, ksize, sigmaX, sigmaY, dst):
        f2D = self.ol.image_filters.filter2D_kernel_f_0
        f2D.rows = src.shape[0]
        f2D.columns = src.shape[1]
        self.filter = "GaussianBlur"
        f2D.mode = 2
        if(sigmaX <= 0):
            sigmaX = 0.3*((ksize[0]-1)*0.5 - 1) + 0.8
        if(sigmaY <= 0):
            sigmaY = sigmaX
        kX = cv2.getGaussianKernel(3,sigmaX,ktype=cv2.CV_32F) #kernel X
        kY = cv2.getGaussianKernel(3,sigmaY,ktype=cv2.CV_32F) #kernel Y
        f2D.r11 = self.floatToFixed(kY[0]*kX[0], 16, 14) + (self.floatToFixed(kY[0]*kX[1], 16, 14) << 16)
        f2D.r12 = self.floatToFixed(kY[0]*kX[2], 16, 14)
        f2D.r21 = self.floatToFixed(kY[1]*kX[0], 16, 14) + (self.floatToFixed(kY[1]*kX[1], 16, 14) << 16)
        f2D.r22 = self.floatToFixed(kY[1]*kX[2], 16, 14)
        f2D.r31 = self.floatToFixed(kY[2]*kX[0], 16, 14) + (self.floatToFixed(kY[2]*kX[1], 16, 14) << 16)
        f2D.r32 = self.floatToFixed(kY[2]*kX[2], 16, 14)
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
           and 'filter2D_kernel_f_0' in description['ip']:
            return True
        return False

class cv2pynqDriverFilter2D(DefaultIP):
    def __init__(self, description):
        super().__init__(description=description)
        
    bindto = ['xilinx.com:hls:filter2D_kernel_f:1.0']

    def start(self):
        self.write(0x00, 0x01)#todo 0x81 restart

    def reset(self):
        self.r11 = 0
        self.r12 = 0
        self.r21 = 0
        self.r22 = 0
        self.r31 = 0
        self.r32 = 0

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
        return self.read(0x44)
    @r2.setter
    def r2(self, value):
        self.write(0x44, value)

    @property
    def r3(self):
        return self.read(0x54)
    @r3.setter
    def r3(self, value):
        self.write(0x54, value)
    
    @property
    def r11(self):
        return self.read(0x34)
    @r11.setter
    def r11(self, value):
        self.write(0x34, value)
    
    @property
    def r12(self):
        return self.read(0x3c)
    @r12.setter
    def r12(self, value):
        self.write(0x3c, value)

    @property
    def r21(self):
        return self.read(0x44)
    @r21.setter
    def r21(self, value):
        self.write(0x44, value)
    
    @property
    def r22(self):
        return self.read(0x4c)
    @r22.setter
    def r22(self, value):
        self.write(0x4c, value)
    
    @property
    def r31(self):
        return self.read(0x54)
    @r31.setter
    def r31(self, value):
        self.write(0x54, value)
    
    @property
    def r32(self):
        return self.read(0x5c)
    @r32.setter
    def r32(self, value):
        self.write(0x5c, value)

    