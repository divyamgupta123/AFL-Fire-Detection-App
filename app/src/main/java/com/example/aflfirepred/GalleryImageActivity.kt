package com.example.aflfirepred

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.media.AudioManager
import android.media.ToneGenerator
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.aflfirepred.databinding.ActivityGalleryImageBinding
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

class GalleryImageActivity : AppCompatActivity(), View.OnClickListener {
    lateinit var binding: ActivityGalleryImageBinding
    lateinit var img: Bitmap

    var interpreter: Interpreter? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityGalleryImageBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.getImageButton.setOnClickListener(this)
        binding.predictImgButton.setOnClickListener(this)
//        interpreter = MainActivity().interpreter!!
        tfModel()
//        interpreter = MainActivity().getInterpreter()
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
                    Toast.makeText(this, "Select Image Now", Toast.LENGTH_SHORT).show()
                }
            }

    }
//    fun predictClass(){
//
//        val model = TfliteMN98acc34epoch.newInstance(applicationContext)
//
//// Creates inputs for reference.
//        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
//
//        val bitmap = Bitmap.createScaledBitmap(img, 224, 224, true)
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

        if (index == 1) {
            ToneGenerator(AudioManager.STREAM_MUSIC, 100).startTone(
                ToneGenerator.TONE_PROP_BEEP,
                5000
            )
        }

        binding.resultText.text = "Label ${label[index]}, Prob:- ${maxProb}"
        Log.e("Gallery", "Label ${label[index]}, Prob:- ${maxProb}")
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (data != null) {
            if (requestCode == 100) {
                binding.imageView.setImageURI(data.data)
                val uri: Uri? = data.data
                try {
                    img = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
                } catch (e: IOException) {
                    Log.e("Main Activity", e.printStackTrace().toString())
                }
            }
        }

    }


    override fun onClick(v: View?) {
        when (v) {
            binding.getImageButton -> {
                Log.e("Main Activity", "Get image called")
                if (interpreter != null) {
                    val intent = Intent(Intent.ACTION_GET_CONTENT)
                    intent.type = "image/*"
                    startActivityForResult(intent, 100)

                } else {
                    Toast.makeText(this, "Model is downloading", Toast.LENGTH_SHORT).show()
                }

            }
            binding.predictImgButton -> {

                Log.e("Main Activity", "Predict called")
                predictClass()
            }
        }
    }

}