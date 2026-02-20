;; Auto-generated. Do not edit!


(when (boundp 'rbo_create::Leds)
  (if (not (find-package "RBO_CREATE"))
    (make-package "RBO_CREATE"))
  (shadow 'Leds (find-package "RBO_CREATE")))
(unless (find-package "RBO_CREATE::LEDS")
  (make-package "RBO_CREATE::LEDS"))
(unless (find-package "RBO_CREATE::LEDSREQUEST")
  (make-package "RBO_CREATE::LEDSREQUEST"))
(unless (find-package "RBO_CREATE::LEDSRESPONSE")
  (make-package "RBO_CREATE::LEDSRESPONSE"))

(in-package "ROS")





(defclass rbo_create::LedsRequest
  :super ros::object
  :slots (_advance _play _color _intensity ))

(defmethod rbo_create::LedsRequest
  (:init
   (&key
    ((:advance __advance) nil)
    ((:play __play) nil)
    ((:color __color) 0)
    ((:intensity __intensity) 0)
    )
   (send-super :init)
   (setq _advance __advance)
   (setq _play __play)
   (setq _color (round __color))
   (setq _intensity (round __intensity))
   self)
  (:advance
   (&optional (__advance :null))
   (if (not (eq __advance :null)) (setq _advance __advance)) _advance)
  (:play
   (&optional (__play :null))
   (if (not (eq __play :null)) (setq _play __play)) _play)
  (:color
   (&optional __color)
   (if __color (setq _color __color)) _color)
  (:intensity
   (&optional __intensity)
   (if __intensity (setq _intensity __intensity)) _intensity)
  (:serialization-length
   ()
   (+
    ;; bool _advance
    1
    ;; bool _play
    1
    ;; uint8 _color
    1
    ;; uint8 _intensity
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _advance
       (if _advance (write-byte -1 s) (write-byte 0 s))
     ;; bool _play
       (if _play (write-byte -1 s) (write-byte 0 s))
     ;; uint8 _color
       (write-byte _color s)
     ;; uint8 _intensity
       (write-byte _intensity s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _advance
     (setq _advance (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _play
     (setq _play (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; uint8 _color
     (setq _color (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _intensity
     (setq _intensity (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;;
   self)
  )

(defclass rbo_create::LedsResponse
  :super ros::object
  :slots (_success ))

(defmethod rbo_create::LedsResponse
  (:init
   (&key
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _success __success)
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass rbo_create::Leds
  :super ros::object
  :slots ())

(setf (get rbo_create::Leds :md5sum-) "926a8fe91d451f4d4bb099ec9e86ff14")
(setf (get rbo_create::Leds :datatype-) "rbo_create/Leds")
(setf (get rbo_create::Leds :request) rbo_create::LedsRequest)
(setf (get rbo_create::Leds :response) rbo_create::LedsResponse)

(defmethod rbo_create::LedsRequest
  (:response () (instance rbo_create::LedsResponse :init)))

(setf (get rbo_create::LedsRequest :md5sum-) "926a8fe91d451f4d4bb099ec9e86ff14")
(setf (get rbo_create::LedsRequest :datatype-) "rbo_create/LedsRequest")
(setf (get rbo_create::LedsRequest :definition-)
      "bool advance
bool play
uint8 color
uint8 intensity
---
bool success

")

(setf (get rbo_create::LedsResponse :md5sum-) "926a8fe91d451f4d4bb099ec9e86ff14")
(setf (get rbo_create::LedsResponse :datatype-) "rbo_create/LedsResponse")
(setf (get rbo_create::LedsResponse :definition-)
      "bool advance
bool play
uint8 color
uint8 intensity
---
bool success

")



(provide :rbo_create/Leds "926a8fe91d451f4d4bb099ec9e86ff14")


