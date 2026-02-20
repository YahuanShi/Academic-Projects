;; Auto-generated. Do not edit!


(when (boundp 'rbo_create::Tank)
  (if (not (find-package "RBO_CREATE"))
    (make-package "RBO_CREATE"))
  (shadow 'Tank (find-package "RBO_CREATE")))
(unless (find-package "RBO_CREATE::TANK")
  (make-package "RBO_CREATE::TANK"))
(unless (find-package "RBO_CREATE::TANKREQUEST")
  (make-package "RBO_CREATE::TANKREQUEST"))
(unless (find-package "RBO_CREATE::TANKRESPONSE")
  (make-package "RBO_CREATE::TANKRESPONSE"))

(in-package "ROS")





(defclass rbo_create::TankRequest
  :super ros::object
  :slots (_left _right ))

(defmethod rbo_create::TankRequest
  (:init
   (&key
    ((:left __left) 0)
    ((:right __right) 0)
    )
   (send-super :init)
   (setq _left (round __left))
   (setq _right (round __right))
   self)
  (:left
   (&optional __left)
   (if __left (setq _left __left)) _left)
  (:right
   (&optional __right)
   (if __right (setq _right __right)) _right)
  (:serialization-length
   ()
   (+
    ;; int16 _left
    2
    ;; int16 _right
    2
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int16 _left
       (write-word _left s)
     ;; int16 _right
       (write-word _right s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int16 _left
     (setq _left (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; int16 _right
     (setq _right (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;;
   self)
  )

(defclass rbo_create::TankResponse
  :super ros::object
  :slots (_success ))

(defmethod rbo_create::TankResponse
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

(defclass rbo_create::Tank
  :super ros::object
  :slots ())

(setf (get rbo_create::Tank :md5sum-) "03e1fb403d16dbbcaf79006d642b45b2")
(setf (get rbo_create::Tank :datatype-) "rbo_create/Tank")
(setf (get rbo_create::Tank :request) rbo_create::TankRequest)
(setf (get rbo_create::Tank :response) rbo_create::TankResponse)

(defmethod rbo_create::TankRequest
  (:response () (instance rbo_create::TankResponse :init)))

(setf (get rbo_create::TankRequest :md5sum-) "03e1fb403d16dbbcaf79006d642b45b2")
(setf (get rbo_create::TankRequest :datatype-) "rbo_create/TankRequest")
(setf (get rbo_create::TankRequest :definition-)
      "int16 left
int16 right
---
bool success

")

(setf (get rbo_create::TankResponse :md5sum-) "03e1fb403d16dbbcaf79006d642b45b2")
(setf (get rbo_create::TankResponse :datatype-) "rbo_create/TankResponse")
(setf (get rbo_create::TankResponse :definition-)
      "int16 left
int16 right
---
bool success

")



(provide :rbo_create/Tank "03e1fb403d16dbbcaf79006d642b45b2")


