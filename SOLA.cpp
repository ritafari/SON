#include <Audio.h>
#define GRANULAR_MEMORY_SIZE 12800  // Adjust size as needed

// Audio objects
AudioInputI2S       micInput;      // I2S microphone input
AudioEffectGranular granular;      // Pitch shifting effect => real-time pitch shifting on Teensy 4.0
AudioOutputI2S      audioOutput;   // I2S output to DAC

// Potentiometer objects
int pitchControlPin = A1;  // Potentiometer connected to analog pin A1
float pitchFactor = 1.0;   // Default pitch shift (1.0 = no change)

// Audio connections (Microphone → Granular Effect → Output)
AudioConnection patchCord1(micInput, 0, granular, 0);
AudioConnection patchCord2(granular, 0, audioOutput, 0);
AudioConnection patchCord3(granular, 0, audioOutput, 1);

AudioControlSGTL5000 audioShield;  // Audio control for Teensy


int16_t granularMemory[GRANULAR_MEMORY_SIZE];

void setup() {
    // Initialize audio hardware
    AudioMemory(12);  // Allocate memory for audio processing
    audioShield.enable();
    audioShield.inputSelect(AUDIO_INPUT_MIC);  // Select microphone input
    audioShield.micGain(30);  // Adjust mic gain as needed

    // Start granular effect (buffer length: 250ms)
    granular.begin(granularMemory, GRANULAR_MEMORY_SIZE);

    // Set initial pitch shift (5 semitones up)
    granular.beginPitchShift(1.3348);  // 2^(5/12) for 5 semitones
}


void loop() {
    // Set mic volume
    int micGainLevel = 20;
    audioShield.micGain(micGainLevel);

    // Read potentiometer (0-4095 for Teensy 4.0)
    int potValue = analogRead(pitchControlPin);
    
    // Map to pitch shift range (0.5x to 2.5x)
    pitchFactor = 0.5 + (potValue / 4095.0) * 2.0;  
    float grainLength = 0.2;  // Grain length in seconds
    
    // Debug: Print pot value and pitch factor
    Serial.print("Potentiometer Value: ");
    Serial.println(potValue);
    
    Serial.print("Mapped Pitch Factor: ");
    Serial.println(pitchFactor);

    // Debug: Indicate when freezing starts
    Serial.println("Starting granular freeze...");
    granular.beginFreeze(grainLength); // Capture audio for processing

    // Debug: Print pitchFactor before applying speed change
    Serial.print("Applying Pitch Shift with Speed Factor: ");
    Serial.println(pitchFactor);
    granular.setSpeed(pitchFactor);  // Apply pitch shifting

    // Short delay (adjust if needed)
    delay(100);  

    // Debug: Indicate when freeze is stopped
    Serial.println("Stopping granular freeze.");
    granular.stop();  // Stop freeze mode when done
}


// If you still want to implement something closer to your Python method, you need to:
  // Manually implement WSOLA in C++ (using circular buffers for overlap-add synthesis).
  // Use fixed-point arithmetic for efficiency.
  // Optimize window placement heuristics (e.g., phase-matching rather than full search).
// Didn't do it here cause best window placement searching using an error-minimization strategy (like in WSOLA) is computationally expensive and impractical for real-time execution on Teensy.
// HOWEVER, check results, if bad we can implement WSOLA in C++

