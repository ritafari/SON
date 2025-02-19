#include <Audio.h>
#define GRANULAR_MEMORY_SIZE 12800

// Audio objects
AudioInputI2S       micInput;
AudioEffectGranular granular;
AudioOutputI2S      audioOutput;

// Potentiometer
int pitchControlPin = A1;
float pitchFactor = 1.0;

// Audio connections
AudioConnection patchCord1(micInput, 0, granular, 0);
AudioConnection patchCord2(granular, 0, audioOutput, 0);
AudioConnection patchCord3(granular, 0, audioOutput, 1);

AudioControlSGTL5000 audioShield;

int16_t granularMemory[GRANULAR_MEMORY_SIZE];

void setup() {
    Serial.begin(9600); // Initialize serial communication
    
    AudioMemory(60); // Increased memory allocation
    audioShield.enable();
    audioShield.inputSelect(AUDIO_INPUT_MIC);
    audioShield.micGain(30);

    granular.begin(granularMemory, GRANULAR_MEMORY_SIZE);
    granular.beginPitchShift(1.0); // Start with no pitch shift
}

void loop() {
    // Read potentiometer and map to pitch factor
    int potValue = analogRead(pitchControlPin);
    pitchFactor = 0.5 + (potValue / 4095.0) * 2.0;
    
    // Apply pitch shift continuously
    granular.setSpeed(pitchFactor);

    // Debug output
    Serial.print("Pitch Factor: ");
    Serial.println(pitchFactor);
    
    delay(50); // Reduced delay for smoother control
}


// If you still want to implement something closer to your Python method, you need to:
  // Manually implement WSOLA in C++ (using circular buffers for overlap-add synthesis).
  // Use fixed-point arithmetic for efficiency.
  // Optimize window placement heuristics (e.g., phase-matching rather than full search).
// Didn't do it here cause best window placement searching using an error-minimization strategy (like in WSOLA) is computationally expensive and impractical for real-time execution on Teensy.
// HOWEVER, check results, if bad we can implement WSOLA in C++

