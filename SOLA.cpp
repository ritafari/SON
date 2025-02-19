
#include <Audio.h>

// Audio objects
AudioInputI2S       micInput;      // I2S microphone input
AudioEffectGranular granular;      // Pitch shifting effect =>  real-time pitch shifting on Teensy 4.0
AudioOutputI2S      audioOutput;   // I2S output to DAC

// Potentiometer objects
int pitchControlPin = A1;  // Potentiometer connected to analog pin A1
float pitchFactor = 1.0;   // Default pitch shift (1.0 = no change)

// Audio connections (Microphone → Granular Effect → Output)
AudioConnection patchCord1(micInput, 0, granular, 0);
AudioConnection patchCord2(granular, 0, audioOutput, 0);
AudioConnection patchCord3(granular, 0, audioOutput, 1);

AudioControlSGTL5000 audioShield;  // Audio control for Teensy

void setup() {
    // Initialize audio hardware
    AudioMemory(12);  // Allocate memory for audio processing
    audioShield.enable();
    audioShield.inputSelect(AUDIO_INPUT_MIC);  // Select microphone input
    audioShield.micGain(20);  // Adjust mic gain as needed

    // Configure granular effect for pitch shifting
    granular.beginPitchShift(1.3348);  // 5 semitone pitch shift (2^(5/12))
}

void loop() {
    // The pitch-shifted audio is processed in real time.
  int potValue = analogRead(pitchControlPin);  // Read potentiometer
    pitchFactor = 0.5 + (potValue / 1023.0) * 2.0;  // Scale to range (0.5x to 2.5x)
    
    granular.setPitch(pitchFactor);  // Apply pitch shift
}


// If you still want to implement something closer to your Python method, you need to:
  // Manually implement WSOLA in C++ (using circular buffers for overlap-add synthesis).
  // Use fixed-point arithmetic for efficiency.
  // Optimize window placement heuristics (e.g., phase-matching rather than full search).
// Didn't do it here cause best window placement searching using an error-minimization strategy (like in WSOLA) is computationally expensive and impractical for real-time execution on Teensy.
// HOWEVER, check results, if bad we can implement WSOLA in C++

