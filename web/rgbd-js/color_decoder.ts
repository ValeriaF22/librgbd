import { NativeYuvFrame, YuvFrame } from './yuv_frame';

export enum ColorCodecType {
  VP8 = 0
}

export class NativeColorDecoder {
  wasmModule: any;
  ptr: number;

  constructor(wasmModule: any, colorCodecType: ColorCodecType) {
    this.wasmModule = wasmModule;
    this.ptr = this.wasmModule.ccall('rgbd_color_decoder_ctor', 'number', ['number'], [colorCodecType]);
  }

  close() {
    this.wasmModule.ccall('rgbd_color_decoder_dtor', null, ['number'], [this.ptr]);
  }

  decode(colorBytes: Uint8Array): YuvFrame {
    const colorBytesPtr = this.wasmModule._malloc(colorBytes.byteLength);
    this.wasmModule.HEAPU8.set(colorBytes, colorBytesPtr);
    const yuvFramePtr = this.wasmModule.ccall('rgbd_color_decoder_decode',
                                              'number',
                                              ['number', 'number', 'number'],
                                              [this.ptr, colorBytesPtr, colorBytes.byteLength]);
    this.wasmModule._free(colorBytesPtr);

    const nativeYuvFrame = new NativeYuvFrame(this.wasmModule, yuvFramePtr);
    const yuvFrame = YuvFrame.fromNative(nativeYuvFrame);
    nativeYuvFrame.close();
    return yuvFrame;
  }
}
