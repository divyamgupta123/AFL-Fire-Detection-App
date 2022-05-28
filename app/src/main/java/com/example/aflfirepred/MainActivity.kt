package com.example.aflfirepred

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.aflfirepred.databinding.ActivityMainBinding
import com.google.firebase.ml.modeldownloader.CustomModel
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions
import com.google.firebase.ml.modeldownloader.DownloadType
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader
import org.tensorflow.lite.Interpreter


class MainActivity : AppCompatActivity(), View.OnClickListener {
    lateinit var binding: ActivityMainBinding
    var interpreter: Interpreter? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.galleryImageButton.setOnClickListener(this)
        binding.liveVideoButton.setOnClickListener(this)
//        tfModel()
    }

    override fun onClick(v: View?) {
        when (v) {
            binding.galleryImageButton -> {

                    startActivity(Intent(this, GalleryImageActivity::class.java))

            }
            binding.liveVideoButton -> {

                    startActivity(Intent(this, LiveVideoActivity::class.java))

            }
        }
    }

    private fun tfModel() {
        val conditions = CustomModelDownloadConditions.Builder()
            .build()
        FirebaseModelDownloader.getInstance()
            .getModel(
                "MN2", DownloadType.LOCAL_MODEL_UPDATE_IN_BACKGROUND,
                conditions
            )
            .addOnSuccessListener { model: CustomModel? ->
                // Download complete. Depending on your app, you could enable the ML
                // feature, or switch from the local model to the remote model, etc.

                // The CustomModel object contains the local path of the model file,
                // which you can use to instantiate a TensorFlow Lite interpreter.
                val modelFile = model?.file
                if (modelFile != null) {
                    interpreter = Interpreter(modelFile)
                }

                Toast.makeText(this, "Model Downloaded", Toast.LENGTH_SHORT).show()
            }

    }

}