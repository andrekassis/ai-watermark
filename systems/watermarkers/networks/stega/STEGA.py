import bchlib
import os
import pickle
from PIL import Image, ImageOps
import numpy as np
import torch
import tensorflow as tf
from tensorflow.compat.v1.saved_model import tag_constants
from tensorflow.compat.v1.saved_model import signature_constants

from watermarkers.networks import BaseWatermarker


class Stega(BaseWatermarker):
    def __init__(
        self,
        checkpoint,
        watermark_path,
        watermark_length,
        image_size=400,
        batch_size=64,
        device="cuda",
    ):
        sess = tf.compat.v1.InteractiveSession(graph=tf.compat.v1.Graph())
        model = tf.compat.v1.saved_model.loader.load(
            sess, [tag_constants.SERVING], checkpoint
        )
        ckpt = (model, sess)

        super().__init__(
            ckpt,
            ckpt,
            watermark_path,
            watermark_length,
            image_size,
            batch_size,
            device,
        )

    def init_watermark(self, watermark_path):
        with open(watermark_path, "rb") as f:
            watermark = pickle.load(f)
        ###
        # return torch.from_numpy(np.array([ ord(w) for w in watermark]).astype(np.float32)).to(self.device).float().view(1, -1)
        return (
            torch.from_numpy(watermark.astype(np.float32))
            .to(self.device)
            .float()
            .view(1, -1)
        )
        ###

    def init_decoder(self, decoder_checkpoint):
        class Decoder:
            def __init__(self, sess, model, image_size):
                self.sess, self.image_size = sess, image_size

                input_image_name = (
                    model.signature_def[
                        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                    ]
                    .inputs["image"]
                    .name
                )
                output_secret_name = (
                    model.signature_def[
                        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                    ]
                    .outputs["decoded"]
                    .name
                )

                self.input_image = tf.compat.v1.get_default_graph().get_tensor_by_name(
                    input_image_name
                )
                self.output_secret = (
                    tf.compat.v1.get_default_graph().get_tensor_by_name(
                        output_secret_name
                    )
                )

                self.bch = bchlib.BCH(137, 5)

            def _extract(self, message):
                ###
                return torch.from_numpy(np.array(message).astype(np.float32))
                ###
                packet_binary = [
                    "".join([str(int(bit)) for bit in secret]) for secret in message
                ]
                packet = [
                    bytes(int(p[i : i + 8], 2) for i in range(0, len(p), 8))
                    for p in packet_binary
                ]
                packet = [bytearray(p) for p in packet]

                codes = []
                for p in packet:
                    bitflips = self.bch.decode_inplace(
                        p[: -self.bch.ecc_bytes], p[-self.bch.ecc_bytes :]
                    )
                    if bitflips != -1:
                        try:
                            code = p[: -self.bch.ecc_bytes].decode("utf-8")
                        except:
                            code = "00000"
                    else:
                        code = "00000"
                    codes.append([ord(c) for c in code][:5])
                return torch.from_numpy(np.array(codes).astype(np.float32))

            def __call__(self, x):
                image = x.permute(0, 2, 3, 1).detach().cpu().numpy()
                decoded = np.array(
                    self.sess.run(
                        [self.output_secret], feed_dict={self.input_image: image}
                    )[0]
                )  # [:,:96]
                # print(decoded)
                return self._extract(decoded).to(x.device)

        model, sess = decoder_checkpoint
        return Decoder(sess, model, self.image_size)

    def init_encoder(self, encoder_checkpoint):
        class Encoder:
            def __init__(self, sess, model, image_size):
                self.sess, self.image_size = sess, image_size

                input_secret_name = (
                    model.signature_def[
                        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                    ]
                    .inputs["secret"]
                    .name
                )
                input_image_name = (
                    model.signature_def[
                        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                    ]
                    .inputs["image"]
                    .name
                )
                output_stegastamp_name = (
                    model.signature_def[
                        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                    ]
                    .outputs["stegastamp"]
                    .name
                )
                output_residual_name = (
                    model.signature_def[
                        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                    ]
                    .outputs["residual"]
                    .name
                )

                self.input_secret = tf.compat.v1.get_default_graph().get_tensor_by_name(
                    input_secret_name
                )
                self.input_image = tf.compat.v1.get_default_graph().get_tensor_by_name(
                    input_image_name
                )
                self.output_stegastamp = (
                    tf.compat.v1.get_default_graph().get_tensor_by_name(
                        output_stegastamp_name
                    )
                )
                self.output_residual = (
                    tf.compat.v1.get_default_graph().get_tensor_by_name(
                        output_residual_name
                    )
                )
                self.bch = bchlib.BCH(137, 5)

            def _make_msg(self, watermark):
                ###
                return watermark.detach().cpu().numpy().astype(np.int32).tolist()
                ###
                watermark = watermark.detach().cpu().numpy().astype(np.int32)
                watermark = ["".join([chr(c) for c in w]) for w in watermark]
                msg = []
                for w in watermark:
                    data = bytearray(w + " " * (7 - len(w)), "utf-8")
                    wmk = [
                        int(xx)
                        for xx in "".join(
                            format(x, "08b") for x in (data + self.bch.encode(data))
                        )
                    ]
                    wmk.extend([0, 0, 0, 0])
                    msg.append(wmk)
                return msg

            def __call__(self, x, msg):
                image = x.permute(0, 2, 3, 1).detach().cpu().numpy()
                hidden_img = self.sess.run(
                    [self.output_stegastamp, self.output_residual],
                    feed_dict={
                        self.input_secret: self._make_msg(msg),
                        self.input_image: image,
                    },
                )[0]
                return (
                    torch.from_numpy(np.array(hidden_img))
                    .to(x.device)
                    .permute(0, 3, 1, 2)
                )

        model, sess = encoder_checkpoint
        return Encoder(sess, model, self.image_size)

    def _encode_batch(self, x_batch, msg_batch):
        return self.encoder(x_batch, msg_batch)

    def _decode_batch(self, x_batch, msg_batch):
        return self._decode_batch_raw(x_batch)

    def _decode_batch_raw(self, x):
        return self.decoder(x)

    def err(self, x_batch, msg_batch):
        return torch.not_equal(x_batch, msg_batch).float().mean(-1)