package com.example.facialexpression;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.media.Image;
import android.util.Log;

import androidx.camera.core.ImageProxy;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.List;
import java.util.Map;

public class Extractor {

    private Context context;
    Interpreter tflite;
    final String ASSOCIATED_AXIS_LABELS = "labels.txt";
    List<String> associatedAxisLabels = null;

    public Extractor(Context context)
    {
        this.context = context;

        try {
            associatedAxisLabels = FileUtil.loadLabels(context, ASSOCIATED_AXIS_LABELS);
        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading label file", e);
        }

        try{
            MappedByteBuffer tfliteModel
                    = FileUtil.loadMappedFile(context,
                    /*"mobilenet_v1_0.25_128_quantized_1_metadata_1.tflite"*/
            "fecnet_model.tflite");
            tflite = new Interpreter(tfliteModel);
        } catch (IOException e){
            Log.e("tfliteSupport", "Error reading model", e);
        }
    }

    public String process(ImageProxy image, boolean useBoxImage){
        @SuppressLint("UnsafeExperimentalUsageError")
        Image img = image.getImage();
        Bitmap bitmap;
        if (useBoxImage) {
            Bitmap bmp = Utils.toBitmap(img);

            int width = bmp.getWidth();
            int height = bmp.getHeight();
            int left, right, top, bottom, diameter;

            diameter = width;
            if (height < width) {
                diameter = height;
            }

            int offset = (int) (0.05 * diameter);
            diameter -= offset;

            left = width / 2 - diameter / 2;
            top = height / 2 - diameter / 2;
            right = width / 2 + diameter / 2;
            bottom = height / 2 + diameter / 2;

            //Log.i("Extractor", "x: " + left + ", y: " + top);
            //Log.i("Extractor", "w: " + (right-left) + ", h: " + (bottom-top));
            //Log.i("Extractor", "w: " + img.getWidth() + ", h: " + img.getHeight());

            bitmap = Bitmap.createBitmap(bmp, left, top, right-left, bottom-top);
        } else {
            bitmap = Utils.toBitmap(img);
        }

        int rotation = Utils.getImageRotation(image);
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        if (true) {
            //facial expression
            int size = height > width ? width : height;
            ImageProcessor imageProcessor = new ImageProcessor.Builder()
                    .add(new ResizeWithCropOrPadOp(size, size))
                    .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                    .add(new Rot90Op(rotation))
                    .build();
            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
            tensorImage.load(bitmap);
            tensorImage = imageProcessor.process(tensorImage);
            TensorBuffer featureBuffer =
                    TensorBuffer.createFixedSize(new int[]{1, 16}, DataType.FLOAT32);
            if (null != tflite) {
                tflite.run(tensorImage.getBuffer(), featureBuffer.getBuffer());
            }

            String result = "[";
            for (int i = 0; i < featureBuffer.getFlatSize(); i++) {
                if (i == featureBuffer.getFlatSize() - 1){
                    result += String.format("%+.4f", featureBuffer.getFloatValue(i)) + "]";
                } else if (i % 4 == 3) {
                    result += String.format("%+.4f", featureBuffer.getFloatValue(i)) + ",\n ";
                } else {
                    result += String.format("%+.4f", featureBuffer.getFloatValue(i)) + ", ";
                }
            }

            return result;
        }
        else {
            int size = height > width ? width : height;
            ImageProcessor imageProcessor = new ImageProcessor.Builder()
                    .add(new ResizeWithCropOrPadOp(size, size))
                    .add(new ResizeOp(128, 128, ResizeOp.ResizeMethod.BILINEAR))
                    .add(new Rot90Op(rotation))
                    .build();
            TensorImage tensorImage = new TensorImage(DataType.UINT8);
            tensorImage.load(bitmap);
            tensorImage = imageProcessor.process(tensorImage);
            TensorBuffer probabilityBuffer =
                    TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
            if (null != tflite) {
                tflite.run(tensorImage.getBuffer(), probabilityBuffer.getBuffer());
            }
            TensorProcessor probabilityProcessor =
                    new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();

            String result = " ";
            if (null != associatedAxisLabels) {
                // Map of labels and their corresponding probability
                TensorLabel labels = new TensorLabel(associatedAxisLabels,
                        probabilityProcessor.process(probabilityBuffer));

                // Create a map to access the result based on label
                Map<String, Float> floatMap = labels.getMapWithFloatValue();
                result = Utils.writeResults(floatMap);
            }

            return result;
        }

    }
}
