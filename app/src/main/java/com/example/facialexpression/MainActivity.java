package com.example.facialexpression;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.support.common.FileUtil;

import java.io.InputStream;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity implements SurfaceHolder.Callback {

    private int REQUEST_CODE_PERMISSIONS = 1001;
    private final String [] REQUIRED_PERMISSIONS = new String []{Manifest.permission.CAMERA};

    PreviewView mPreviewView;
    TextView tvResults;
    SurfaceHolder holder;
    SurfaceView surfaceView;
    Canvas canvas;
    Paint paint;
    Extractor extractor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mPreviewView = findViewById(R.id.viewFinder);
        tvResults = findViewById(R.id.tvResults);

        extractor = new Extractor(this);

        // CameraX
        if (allPermissionsGranted()){
            startCamera();
        }
        else{
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS,
                    REQUEST_CODE_PERMISSIONS);
        }

        //Create the bounding box
        surfaceView = findViewById(R.id.overlay);
        surfaceView.setZOrderOnTop(true);
        holder = surfaceView.getHolder();
        holder.setFormat(PixelFormat.TRANSPARENT);
        holder.addCallback(this);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        //test : load image
        try {
            byte[] sample = FileUtil.loadByteFromFile(this, "face.jpg");
            Bitmap bitmap = BitmapFactory.decodeByteArray(sample, 0, sample.length);
            Bitmap rzBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

            //Matrix matrix = new Matrix();
            //matrix.postRotate(90);
            //Bitmap rtBitmap = Bitmap.createBitmap(rzBitmap, 0, 0 , rzBitmap.getWidth(), rzBitmap.getHeight(), matrix, true);

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

            Log.i("rzbmp", "[" + rzBitmap.getWidth() + ", " + rzBitmap.getHeight() + "]");
            for (int y = 0; y < 224; y++) { //h
                Log.i("rzbmp", "[" + "0," + y + ":" + String.format("%.5f, %.5f, %.5f",
                        inputBuffer[0][0][y][0], inputBuffer[0][0][y][1], inputBuffer[0][0][y][2]) + "]");
            }

            k = 0;
            for (int y = 0; y < 224; y++) { //h
                for (int x = 0; x < 224; x++) { //w
                    int r = (int)(inputBuffer[0][0][y][x] * 255.0f + 0.5f);
                    int g = (int)(inputBuffer[0][1][y][x] * 255.0f + 0.5f);
                    int b = (int)(inputBuffer[0][2][y][x] * 255.0f + 0.5f);

                    pixels[k] = 0;
                    pixels[k++] = ((r & 0xff) << 16) | ((g & 0xff) << 8) | ((b & 0xff));
                }
            }

            rzBitmap.setPixels(pixels, 0, rzBitmap.getWidth(), 0,0,224,224);

            ImageView imageView = (ImageView) findViewById(R.id.image1);
            imageView.setImageBitmap(rzBitmap);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void startCamera(){
        ListenableFuture<ProcessCameraProvider>
                cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(
                new Runnable() {
                    @Override
                    public void run() {
                        try {
                            ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                            bindPreview(cameraProvider);
                        } catch (ExecutionException | InterruptedException e) { }
                    }
                },
                ActivityCompat.getMainExecutor(this)
        );
    }

    void bindPreview(ProcessCameraProvider cameraProvider) {
        ImageCapture.Builder builder = new ImageCapture.Builder();
        ImageCapture imageCapture = builder.build();

        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(mPreviewView.createSurfaceProvider());

        CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(
                CameraSelector.LENS_FACING_FRONT).build();

        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();
        imageAnalysis.setAnalyzer(ActivityCompat.getMainExecutor(this),
                new ImageAnalysis.Analyzer() {
                    @Override
                    public void analyze(@NonNull ImageProxy image) {
                        String result = "hello";
                        result = extractor.process(image, 1);
                        tvResults.setText(result);
                        image.close();
                    }
                });

        Camera camera = cameraProvider.bindToLifecycle(this, cameraSelector,
                preview, imageAnalysis, imageCapture);
    }

    private boolean allPermissionsGranted(){
        for(String permission: REQUIRED_PERMISSIONS){
            if(ContextCompat.checkSelfPermission(
                    this, permission) != PackageManager.PERMISSION_GRANTED){
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CODE_PERMISSIONS){
            if(allPermissionsGranted()){
                startCamera();
            }
            else{
                Toast.makeText(this, "Please, give access to camera",
                        Toast.LENGTH_SHORT).show();
                this.finish();
            }
        }
    }

    // For drawing the rectangular box
    private void DrawFocusRect(int color) {
        //DisplayMetrics displaymetrics = new DisplayMetrics();
        //getWindowManager().getDefaultDisplay().getMetrics(displaymetrics);
        int height = mPreviewView.getHeight();
        int width = mPreviewView.getWidth();
        int left, right, top, bottom, diameter;

        //Log.i("Preview", "x: " + mPreviewView.getLeft() + ", y: " + mPreviewView.getTop());
        //Log.i("Preview", "w: " + width + ", h: " + height);

        diameter = width;
        if (height < width) {
            diameter = height;
        }

        int offset = (int) (0.05 * diameter);
        diameter -= offset;

        //Changing the value of x in diameter/x will change the size of the box ; inversely proportionate to x
        left = width / 2 - diameter / 2;
        top = height / 2 - diameter / 2;
        right = width / 2 + diameter / 2;
        bottom = height / 2 + diameter / 2;

        //border's properties
        paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setColor(color);
        paint.setStrokeWidth(5);

        canvas = holder.lockCanvas();
        canvas.drawColor(0, PorterDuff.Mode.CLEAR);
        canvas.drawRect(left, top, right, bottom, paint);
        holder.unlockCanvasAndPost(canvas);
    }

    // Callback functions for the surface Holder
    @Override
    public void surfaceCreated(SurfaceHolder holder) {

    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        //Drawing rectangle
        DrawFocusRect(Color.parseColor("#b3dabb"));
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {

    }
}