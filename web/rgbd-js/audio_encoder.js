import { NativeByteArray } from './capi_containers.js';

export class AudioEncoderFrame {
  constructor(nativeFrame) {
    let packetBytesList = [];
    const packetBytesListCount = nativeFrame.getPacketBytesListCount();
    for (let i = 0; i < packetBytesListCount; ++i) {
      const packetBytes = nativeFrame.getPacketBytes(i);
      packetBytesList.push(packetBytes);
    }
    this.packetBytesList = packetBytesList;
  }
}

class NativeAudioEncoderFrame {
  constructor(wasmModule, ptr) {
    this.wasmModule = wasmModule;
    this.ptr = ptr;
  }

  close() {
    this.wasmModule.ccall('rgbd_audio_encoder_frame_dtor', null, ['number'], [this.ptr]);
  }

  getPacketBytesListCount() {
    return this.wasmModule.ccall('rgbd_audio_encoder_frame_get_packet_bytes_list_count',
                                 'number',
                                 ['number'],
                                 [this.ptr]);
  }

  getPacketBytes(index) {
    const byteArrayPtr = this.wasmModule.ccall('rgbd_audio_encoder_frame_get_packet_bytes',
                                               'number',
                                               ['number', 'number'],
                                               [this.ptr, index]);
    const byteArray = new NativeByteArray(this.wasmModule, byteArrayPtr);
    const bytes = byteArray.toArray();
    byteArray.close();
    return bytes;
  }
}

export class NativeAudioEncoder {
  constructor(wasmModule) {
    this.wasmModule = wasmModule;
    this.ptr = this.wasmModule.ccall('rgbd_audio_encoder_ctor', 'number', [], []);
  }

  close() {
    this.wasmModule.ccall('rgbd_audio_encoder_dtor', null, ['number'], [this.ptr]);
  }

  encode(pcmSamples) {
    const pcmSamplesPtr = this.wasmModule._malloc(pcmSamples.byteLength);
    this.wasmModule.HEAPU8.set(pcmSamples, pcmSamplesPtr);
    const framePtr = this.wasmModule.ccall('rgbd_audio_encoder_encode',
                                           'number',
                                           ['number', 'number', 'number'],
                                           [this.ptr, pcmSamplesPtr, pcmSamples.length]);
    this.wasmModule._free(pcmSamplesPtr);

    const nativeFrame = new NativeAudioEncoderFrame(this.wasmModule, framePtr);
    const frame = new AudioEncoderFrame(nativeFrame);
    nativeFrame.close();
    return frame;
  }

  flush() {
    const framePtr = this.wasmModule.ccall('rgbd_audio_encoder_flush',
                                           'number',
                                           ['number'],
                                           [this.ptr]);

    const nativeFrame = new NativeAudioEncoderFrame(this.wasmModule, framePtr);
    const frame = new AudioEncoderFrame(nativeFrame);
    nativeFrame.close();
    return frame;
  }
}
