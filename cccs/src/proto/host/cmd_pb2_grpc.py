# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import cmd_pb2 as cmd__pb2


class CompressServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RunCompress = channel.unary_stream(
                '/CompressService/RunCompress',
                request_serializer=cmd__pb2.CompressCmd.SerializeToString,
                response_deserializer=cmd__pb2.CompressResult.FromString,
                )
        self.setImage = channel.unary_unary(
                '/CompressService/setImage',
                request_serializer=cmd__pb2.ImageClass.SerializeToString,
                response_deserializer=cmd__pb2.Empty.FromString,
                )
        self.setContinue = channel.unary_unary(
                '/CompressService/setContinue',
                request_serializer=cmd__pb2.Empty.SerializeToString,
                response_deserializer=cmd__pb2.Empty.FromString,
                )
        self.setNoContinue = channel.unary_unary(
                '/CompressService/setNoContinue',
                request_serializer=cmd__pb2.Empty.SerializeToString,
                response_deserializer=cmd__pb2.Empty.FromString,
                )
        self.getMonitorResult = channel.unary_unary(
                '/CompressService/getMonitorResult',
                request_serializer=cmd__pb2.Empty.SerializeToString,
                response_deserializer=cmd__pb2.YoloResponse.FromString,
                )
        self.getMonitorImg = channel.unary_unary(
                '/CompressService/getMonitorImg',
                request_serializer=cmd__pb2.Empty.SerializeToString,
                response_deserializer=cmd__pb2.YoloImg.FromString,
                )
        self.setDevice = channel.unary_unary(
                '/CompressService/setDevice',
                request_serializer=cmd__pb2.CmpDevice.SerializeToString,
                response_deserializer=cmd__pb2.Empty.FromString,
                )
        self.setPYNQ = channel.unary_unary(
                '/CompressService/setPYNQ',
                request_serializer=cmd__pb2.PYNQEnableCmd.SerializeToString,
                response_deserializer=cmd__pb2.Empty.FromString,
                )
        self.uploadPYNQ = channel.unary_unary(
                '/CompressService/uploadPYNQ',
                request_serializer=cmd__pb2.PYNQBitsCmd.SerializeToString,
                response_deserializer=cmd__pb2.Empty.FromString,
                )
        self.uploadLattice = channel.unary_unary(
                '/CompressService/uploadLattice',
                request_serializer=cmd__pb2.LatticeBitsCmd.SerializeToString,
                response_deserializer=cmd__pb2.Empty.FromString,
                )
        self.setMotor = channel.unary_unary(
                '/CompressService/setMotor',
                request_serializer=cmd__pb2.MotorCmd.SerializeToString,
                response_deserializer=cmd__pb2.Empty.FromString,
                )


class CompressServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def RunCompress(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def setImage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def setContinue(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def setNoContinue(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getMonitorResult(self, request, context):
        """ultra96
        rpc enableAlwaysMonitor(Empty) returns (Empty);
        rpc disableAlwaysMonitor(Empty) returns (Empty);
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getMonitorImg(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def setDevice(self, request, context):
        """set cpu or pynq
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def setPYNQ(self, request, context):
        """PYNQ
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def uploadPYNQ(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def uploadLattice(self, request, context):
        """Lattice
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def setMotor(self, request, context):
        """motro
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CompressServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'RunCompress': grpc.unary_stream_rpc_method_handler(
                    servicer.RunCompress,
                    request_deserializer=cmd__pb2.CompressCmd.FromString,
                    response_serializer=cmd__pb2.CompressResult.SerializeToString,
            ),
            'setImage': grpc.unary_unary_rpc_method_handler(
                    servicer.setImage,
                    request_deserializer=cmd__pb2.ImageClass.FromString,
                    response_serializer=cmd__pb2.Empty.SerializeToString,
            ),
            'setContinue': grpc.unary_unary_rpc_method_handler(
                    servicer.setContinue,
                    request_deserializer=cmd__pb2.Empty.FromString,
                    response_serializer=cmd__pb2.Empty.SerializeToString,
            ),
            'setNoContinue': grpc.unary_unary_rpc_method_handler(
                    servicer.setNoContinue,
                    request_deserializer=cmd__pb2.Empty.FromString,
                    response_serializer=cmd__pb2.Empty.SerializeToString,
            ),
            'getMonitorResult': grpc.unary_unary_rpc_method_handler(
                    servicer.getMonitorResult,
                    request_deserializer=cmd__pb2.Empty.FromString,
                    response_serializer=cmd__pb2.YoloResponse.SerializeToString,
            ),
            'getMonitorImg': grpc.unary_unary_rpc_method_handler(
                    servicer.getMonitorImg,
                    request_deserializer=cmd__pb2.Empty.FromString,
                    response_serializer=cmd__pb2.YoloImg.SerializeToString,
            ),
            'setDevice': grpc.unary_unary_rpc_method_handler(
                    servicer.setDevice,
                    request_deserializer=cmd__pb2.CmpDevice.FromString,
                    response_serializer=cmd__pb2.Empty.SerializeToString,
            ),
            'setPYNQ': grpc.unary_unary_rpc_method_handler(
                    servicer.setPYNQ,
                    request_deserializer=cmd__pb2.PYNQEnableCmd.FromString,
                    response_serializer=cmd__pb2.Empty.SerializeToString,
            ),
            'uploadPYNQ': grpc.unary_unary_rpc_method_handler(
                    servicer.uploadPYNQ,
                    request_deserializer=cmd__pb2.PYNQBitsCmd.FromString,
                    response_serializer=cmd__pb2.Empty.SerializeToString,
            ),
            'uploadLattice': grpc.unary_unary_rpc_method_handler(
                    servicer.uploadLattice,
                    request_deserializer=cmd__pb2.LatticeBitsCmd.FromString,
                    response_serializer=cmd__pb2.Empty.SerializeToString,
            ),
            'setMotor': grpc.unary_unary_rpc_method_handler(
                    servicer.setMotor,
                    request_deserializer=cmd__pb2.MotorCmd.FromString,
                    response_serializer=cmd__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'CompressService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class CompressService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RunCompress(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/CompressService/RunCompress',
            cmd__pb2.CompressCmd.SerializeToString,
            cmd__pb2.CompressResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def setImage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CompressService/setImage',
            cmd__pb2.ImageClass.SerializeToString,
            cmd__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def setContinue(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CompressService/setContinue',
            cmd__pb2.Empty.SerializeToString,
            cmd__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def setNoContinue(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CompressService/setNoContinue',
            cmd__pb2.Empty.SerializeToString,
            cmd__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getMonitorResult(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CompressService/getMonitorResult',
            cmd__pb2.Empty.SerializeToString,
            cmd__pb2.YoloResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getMonitorImg(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CompressService/getMonitorImg',
            cmd__pb2.Empty.SerializeToString,
            cmd__pb2.YoloImg.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def setDevice(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CompressService/setDevice',
            cmd__pb2.CmpDevice.SerializeToString,
            cmd__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def setPYNQ(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CompressService/setPYNQ',
            cmd__pb2.PYNQEnableCmd.SerializeToString,
            cmd__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def uploadPYNQ(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CompressService/uploadPYNQ',
            cmd__pb2.PYNQBitsCmd.SerializeToString,
            cmd__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def uploadLattice(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CompressService/uploadLattice',
            cmd__pb2.LatticeBitsCmd.SerializeToString,
            cmd__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def setMotor(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CompressService/setMotor',
            cmd__pb2.MotorCmd.SerializeToString,
            cmd__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class Yolov3SeviceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.runyolov3 = channel.unary_unary(
                '/Yolov3Sevice/runyolov3',
                request_serializer=cmd__pb2.Yolov3Datain.SerializeToString,
                response_deserializer=cmd__pb2.YoloResponse.FromString,
                )
        self.getImage = channel.unary_unary(
                '/Yolov3Sevice/getImage',
                request_serializer=cmd__pb2.Empty.SerializeToString,
                response_deserializer=cmd__pb2.YoloImg.FromString,
                )


class Yolov3SeviceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def runyolov3(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getImage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_Yolov3SeviceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'runyolov3': grpc.unary_unary_rpc_method_handler(
                    servicer.runyolov3,
                    request_deserializer=cmd__pb2.Yolov3Datain.FromString,
                    response_serializer=cmd__pb2.YoloResponse.SerializeToString,
            ),
            'getImage': grpc.unary_unary_rpc_method_handler(
                    servicer.getImage,
                    request_deserializer=cmd__pb2.Empty.FromString,
                    response_serializer=cmd__pb2.YoloImg.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Yolov3Sevice', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Yolov3Sevice(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def runyolov3(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Yolov3Sevice/runyolov3',
            cmd__pb2.Yolov3Datain.SerializeToString,
            cmd__pb2.YoloResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getImage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Yolov3Sevice/getImage',
            cmd__pb2.Empty.SerializeToString,
            cmd__pb2.YoloImg.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
