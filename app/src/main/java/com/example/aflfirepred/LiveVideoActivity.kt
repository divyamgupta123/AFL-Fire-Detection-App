package com.example.aflfirepred

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ImageFormat
import android.media.Image
import android.os.Bundle
import android.renderscript.*
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.aflfirepred.databinding.ActivityLiveVideoBinding
import com.google.firebase.ml.modeldownloader.CustomModel
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions
import com.google.firebase.ml.modeldownloader.DownloadType
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class LiveVideoActivity : AppCompatActivity(), ImageAnalysis.Analyzer {
    lateinit var binding: ActivityLiveVideoBinding

    private lateinit var cameraExecutor: ExecutorService

    lateinit var img: Bitmap
    var interpreter: Interpreter? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityLiveVideoBinding.inflate(layoutInflater)
        setContentView(binding.root)

        tfModel()

        // Request camera permissions
        if (allPermissionsGranted()) {
            if (interpreter != null) {
                startCamera()
            } else {
                Toast.makeText(this, "Model is downloading", Toast.LENGTH_SHORT).show()
            }
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()

            preview.setSurfaceProvider(binding.viewFinder.surfaceProvider)

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            val imgAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imgAnalyzer.setAnalyzer(
                cameraExecutor,
                ImageAnalysis.Analyzer() { image -> this.analyze(image) })

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, imgAnalyzer, preview
                )

            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))

    }


    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }


    @SuppressLint("UnsafeOptInUsageError")
    override fun analyze(image: ImageProxy) {
        Log.e(TAG, "Ananlyse Called")

        val mediaImg = image.image
        if (mediaImg != null) {

            img = yuv420ToBitmap(mediaImg, this)
            predictClass()
            image.close()
        }
    }

    private fun tfModel() {
        val conditions = CustomModelDownloadConditions.Builder()
            .build()
        FirebaseModelDownloader.getInstance()
            .getModel(
                "SN98", DownloadType.LOCAL_MODEL_UPDATE_IN_BACKGROUND,
                conditions
            )
            .addOnSuccessListener { model: CustomModel? ->
                // Download complete. Depending on your app, you could enable the ML
                // feature, or switch from the local model to the remote model, etc.

                // The CustomModel object contains the local path of the model file,
                // which you can use to instantiate a TensorFlow Lite interpreter.
                Log.e("Main Activity", "Model Downloaded")
                val modelFile = model?.file
                if (modelFile != null) {
                    interpreter = Interpreter(modelFile)
                    startCamera()
                }

            }

    }

    private fun predictClass() {
        val bitmap = Bitmap.createScaledBitmap(img, 227, 227, true)
        val input = ByteBuffer.allocateDirect(227 * 227 * 3 * 4).order(ByteOrder.nativeOrder())
        for (y in 0 until 227) {
            for (x in 0 until 227) {
                val px = bitmap.getPixel(x, y)

                // Get channel values from the pixel value.
                val r = Color.red(px)
                val g = Color.green(px)
                val b = Color.blue(px)

                // Normalize channel values to [-1.0, 1.0]. This requirement depends on the model.
                // For example, some models might require values to be normalized to the range
                // [0.0, 1.0] instead.
                val rf = r / 255f
                val gf = g / 255f
                val bf = b / 255f

                input.putFloat(rf)
                input.putFloat(gf)
                input.putFloat(bf)
            }
        }

        val bufferSize = 2 * java.lang.Float.SIZE / java.lang.Byte.SIZE
        val modelOutput = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder())
        interpreter?.run(input, modelOutput)

        modelOutput.rewind()
        val probabilities = modelOutput.asFloatBuffer()
        var label: ArrayList<String> = ArrayList()
        var probs: ArrayList<Float> = ArrayList()
        try {
            val reader = BufferedReader(
                InputStreamReader(assets.open("class_labels.txt"))
            )
            for (i in 0 until probabilities.capacity()) {

                label.add(reader.readLine())
                probs.add(probabilities[i])

            }
        } catch (e: IOException) {
            // File not found?
        }

        val maxProb = probs.maxOrNull() ?: 0
        val index = probs.indexOf(maxProb)

//        if (index == 1) {
//            ToneGenerator(AudioManager.STREAM_MUSIC, 100).startTone(
//                ToneGenerator.TONE_PROP_BEEP,
//                5000
//            )
//        }

        binding.resultText.text = "Label ${label[index]}, Prob:- ${maxProb}"
    }

//    fun predictClass(){
//
//        val model = TfliteMN98acc34epoch.newInstance(applicationContext)
//
//        val bitmap = Bitmap.createScaledBitmap(img, 224, 224, true)
//// Creates inputs for reference.
//        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
//
//        val tensorImage: TensorImage = TensorImage(DataType.FLOAT32)
//        tensorImage.load(bitmap)
//        val byteBuffer:ByteBuffer = tensorImage.buffer
//
//        inputFeature0.loadBuffer(byteBuffer)
//
//// Runs model inference and gets result.
//        val outputs = model.process(inputFeature0)
//        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
//
//        binding.resultText.text = "${outputFeature0.floatArray[0]}   ${outputFeature0.floatArray[1]}  ${outputFeature0.floatArray[2]}"
//// Releases model resources if no longer used.
//        model.close()
//    }

    private fun yuv420ToBitmap(image: Image, context: Context): Bitmap {
        val rs = RenderScript.create(this)
        val script = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))

        // Refer the logic in a section below on how to convert a YUV_420_888 image
        // to single channel flat 1D array. For sake of this example I'll abstract it
        // as a method.
        val yuvByteArray: ByteArray = image2byteArray(image)!!
        val yuvType: Type.Builder = Type.Builder(rs, Element.U8(rs)).setX(yuvByteArray.size)
        val `in` = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT)
        val rgbaType: Type.Builder = Type.Builder(rs, Element.RGBA_8888(rs))
            .setX(image.getWidth())
            .setY(image.getHeight())
        val out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT)

        // The allocations above "should" be cached if you are going to perform
        // repeated conversion of YUV_420_888 to Bitmap.
        `in`.copyFrom(yuvByteArray)
        script.setInput(`in`)
        script.forEach(out)
        val bitmap =
            Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888)
        out.copyTo(bitmap)
        return bitmap
    }

    private fun image2byteArray(image: Image): ByteArray? {
        require(image.format == ImageFormat.YUV_420_888) { "Invalid image format" }
        val width = image.width
        val height = image.height
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        // Full size Y channel and quarter size U+V channels.
        val numPixels = (width * height * 1.5f).toInt()
        val nv21 = ByteArray(numPixels)
        var index = 0

        // Copy Y channel.
        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        for (y in 0 until height) {
            for (x in 0 until width) {
                nv21[index++] = yBuffer[y * yRowStride + x * yPixelStride]
            }
        }

        // Copy VU data; NV21 format is expected to have YYYYVU packaging.
        // The U/V planes are guaranteed to have the same row stride and pixel stride.
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride
        val uvWidth = width / 2
        val uvHeight = height / 2
        for (y in 0 until uvHeight) {
            for (x in 0 until uvWidth) {
                val bufferIndex = y * uvRowStride + x * uvPixelStride
                // V channel.
                nv21[index++] = vBuffer[bufferIndex]
                // U channel.
                nv21[index++] = uBuffer[bufferIndex]
            }
        }
        return nv21
    }

    companion object {
        private const val TAG = "CameraXBasic"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
