package com.example.facialexpression;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.util.List;
import java.util.Map;

public class FeExtractor {

    private Context context;
    Interpreter tflite;

    public FeExtractor(Context context)
    {
        this.context = context;

        // load tflite model for facial expression extraction
        try{
            MappedByteBuffer tfliteModel
                    = FileUtil.loadMappedFile(context, "fecnet_model.tflite");
            tflite = new Interpreter(tfliteModel);
        } catch (IOException e){
            Log.e("tfliteSupport", "Error reading model", e);
        }
    }

    public String process(ImageProxy image, int imageType){
        @SuppressLint("UnsafeExperimentalUsageError")
        Image img = image.getImage();
        Bitmap bitmap;
        if (imageType == 0) { //use cropped image
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
        } else { //use full image
            bitmap = Utils.toBitmap(img);
        }

        //facial expression
        //bitmap [1, 224, 224, 3] -> tensor [1, 3, 224, 224]
        Bitmap rzBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

        int[] pixels = new int[224 * 224];
        rzBitmap.getPixels(pixels, 0, rzBitmap.getWidth(), 0, 0, 224, 224);

        float[][][][] inputBuffer = new float[1][3][224][224];
        int k = 0;
        for (int y = 0; y < 224; y++) { //h
            for (int x = 0; x < 224; x++) { //w
                int pixel = pixels[k++];
                inputBuffer[0][0][y][x] = ((pixel >> 16) & 0xff) / 255.0f; //r
                inputBuffer[0][1][y][x] = ((pixel >> 8) & 0xff) / 255.0f;  //g
                inputBuffer[0][2][y][x] = ((pixel >> 0) & 0xff) / 255.0f;  //b
            }
        }

        //[1, 16]
        TensorBuffer featureBuffer =
                TensorBuffer.createFixedSize(new int[]{1, 16}, DataType.FLOAT32);

        //inference
        if (null != tflite) {
            tflite.run(inputBuffer, featureBuffer.getBuffer());
        }

        //result
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
}
