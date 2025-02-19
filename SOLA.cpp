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
    Serial.begin(9600);
    AudioMemory(80);  // Increased memory

    // Enable audio shield
    audioShield.enable();
    audioShield.inputSelect(AUDIO_INPUT_MIC);
    audioShield.micGain(30);  // Increase mic gain

    // Set output volume
    audioShield.volume(0.5);
    audioShield.unmuteHeadphone();
    audioShield.unmuteLineout();

    granular.begin(granularMemory, GRANULAR_MEMORY_SIZE);
    granular.beginPitchShift(1.0);  // Start with no pitch shift

    Serial.println("Setup complete.");
}

void loop() {
    // Read potentiometer and map to pitch range (0.5x to 2.0x)
    int potValue = analogRead(pitchControlPin);
    pitchFactor = map(potValue, 0, 1023, 50, 200) / 100.0;  // Range: 0.5x to 2.0x

    // Apply pitch shift without restarting the effect
    granular.setSpeed(pitchFactor);

    // Debugging output
    Serial.print("Pitch Factor: ");
    Serial.println(pitchFactor);

    delay(50);
}



// If you still want to implement something closer to your Python method, you need to:
  // Manually implement WSOLA in C++ (using circular buffers for overlap-add synthesis).
  // Use fixed-point arithmetic for efficiency.
  // Optimize window placement heuristics (e.g., phase-matching rather than full search).
// Didn't do it here cause best window placement searching using an error-minimization strategy (like in WSOLA) is computationally expensive and impractical for real-time execution on Teensy.
// HOWEVER, check results, if bad we can implement WSOLA in C++

