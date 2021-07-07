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
    final String ASSOCIATED_AXIS_LABELS = "labels.txt";
    List<String> associatedAxisLabels = null;
    private byte[] sample;
    private float[][][][] txtInputs = new float[1][3][224][224];

    public FeExtractor(Context context)
    {
        this.context = context;

        try {
            associatedAxisLabels = FileUtil.loadLabels(context, ASSOCIATED_AXIS_LABELS);
        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading label file", e);
        }

        try{
            MappedByteBuffer tfliteModel
                    = FileUtil.loadMappedFile(context, "fecnet_model.tflite");
            tflite = new Interpreter(tfliteModel);
        } catch (IOException e){
            Log.e("tfliteSupport", "Error reading model", e);
        }

        try{
            sample = FileUtil.loadByteFromFile(context, "face.jpg");
        } catch (IOException e){
            Log.e("sampleRead", "Error reading sample file", e);
        }

        //load text file
        StringBuffer strBuffer = new StringBuffer();
        try{
            InputStream is = context.getAssets().open("dump_face.csv");
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));

            int c = 0, h = 0;
            String line="";
            while((line=reader.readLine())!=null){
                strBuffer.append(line+"\n");
                String[] temp = line.split(",");
                for (int w = 0 ; w < temp.length; w++) {
                    txtInputs[0][c][h][w] = Float.parseFloat(temp[w]);
                    //Log.i("test", String.format("%.5f", f));
                }
                h++;
                if (h == 224) {
                    h = 0;
                    c++;
                }
            }
            reader.close();
            is.close();
        }catch (IOException e){
            e.printStackTrace();
        }
    }

    public String process(ImageProxy image, int imageType){
        @SuppressLint("UnsafeExperimentalUsageError")
        Image img = image.getImage();
        Bitmap bitmap;
        if (imageType == 0) {
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
        } else if (imageType == 1) {
            bitmap = BitmapFactory.decodeByteArray(sample, 0, sample.length);
        } else {
            bitmap = Utils.toBitmap(img);
        }

        int rotation = Utils.getImageRotation(image);
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        Log.i("test1", "[" + width + ", " + height + "]");

        if (true) {
            //facial expression
            int size = height > width ? width : height;
            ImageProcessor imageProcessor = new ImageProcessor.Builder()
                    //.add(new ResizeWithCropOrPadOp(size, size))
                    .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                    .add(new NormalizeOp(0, 255))
                    //.add(new Rot90Op(rotation))
                    .build();

            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
            tensorImage.load(bitmap);
            tensorImage = imageProcessor.process(tensorImage);

            //[1, 3, 224, 224]
            //int[] imageShape = tflite.getInputTensor(0).shape();
            //Log.i("test1", "[" + imageShape[0] + ", " + imageShape[1] + ", " + imageShape[2] + ", " + imageShape[3] + "]");
            //Log.i("test2", "[" + imageShape[0] + ", " + tensorImage.getColorSpaceType() + ", " + tensorImage.getWidth() + ", " + tensorImage.getHeight() + "]");

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

            /*
            Log.i("rzbmp", "[" + rzBitmap.getWidth() + ", " + rzBitmap.getHeight() + "]");
            for (int y = 0; y < 224; y++) { //h
                Log.i("rzbmp", "[" + "0," + y + ":" + String.format("%.5f, %.5f, %.5f",
                        inputBuffer[0][0][y][0], inputBuffer[0][0][y][1], inputBuffer[0][0][y][2]) + "]");
            }
            */

            //Log.i("test1", "[" + inputBuffer[0][0][127][127] + ", " + inputBuffer[0][0][127][128] + "]");
            //Log.i("test1", "[" + inputBuffer[0][0][128][127] + ", " + inputBuffer[0][0][128][128] + "]");
            //Log.i("test1", "[" + inputBuffer[0][1][127][127] + ", " + inputBuffer[0][1][127][128] + "]");
            //Log.i("test1", "[" + inputBuffer[0][1][128][127] + ", " + inputBuffer[0][1][128][128] + "]");
            //Log.i("test1", "[" + inputBuffer[0][2][127][127] + ", " + inputBuffer[0][2][127][128] + "]");
            //Log.i("test1", "[" + inputBuffer[0][2][128][127] + ", " + inputBuffer[0][2][128][128] + "]");

            //[1, 16]
            TensorBuffer featureBuffer =
                    TensorBuffer.createFixedSize(new int[]{1, 16}, DataType.FLOAT32);

            if (null != tflite) {
                //tflite.run(tensorImage.getBuffer(), featureBuffer.getBuffer());
                tflite.run(inputBuffer, featureBuffer.getBuffer());
                //tflite.run(txtInputs, featureBuffer.getBuffer());
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
