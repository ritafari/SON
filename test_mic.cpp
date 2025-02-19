#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>


AudioInputI2S            micInput;  
AudioOutputI2S           headphonesOut;  
AudioConnection          patchCord1(micInput, 0, headphonesOut, 0); 
AudioConnection          patchCord2(micInput, 1, headphonesOut, 1); 
AudioControlSGTL5000     audioShield;    

void setup() {
    AudioMemory(10);    
    audioShield.enable(); 
    audioShield.inputSelect(AUDIO_INPUT_MIC);  
    audioShield.volume(0.5);
    audioShield.micGain(30);  
}

void loop() {

    int micGainLevel = 20;
    audioShield.micGain(micGainLevel);

    delay(100);
}
