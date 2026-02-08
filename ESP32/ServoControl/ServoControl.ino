#include <ESP32Servo.h>

Servo servo;

void setup() 
{

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  
  servo.setPeriodHertz(50);    // Standard 50hz servo
  servo.attach(12, 500, 2400);

  Serial.begin(115200);

}

void loop() 
{
  servo.write(45);
  delay(1000);
  servo.write(135);
  delay(1000);
}
