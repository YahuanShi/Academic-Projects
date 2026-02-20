;; Auto-generated. Do not edit!


(when (boundp 'rbo_create::ResetOdom)
  (if (not (find-package "RBO_CREATE"))
    (make-package "RBO_CREATE"))
  (shadow 'ResetOdom (find-package "RBO_CREATE")))
(unless (find-package "RBO_CREATE::RESETODOM")
  (make-package "RBO_CREATE::RESETODOM"))
(unless (find-package "RBO_CREATE::RESETODOMREQUEST")
  (make-package "RBO_CREATE::RESETODOMREQUEST"))
(unless (find-package "RBO_CREATE::RESETODOMRESPONSE")
  (make-package "RBO_CREATE::RESETODOMRESPONSE"))

(in-package "ROS")





(defclass rbo_create::ResetOdomRequest
  :super ros::object
  :slots (_reset_odom ))

(defmethod rbo_create::ResetOdomRequest
  (:init
   (&key
    ((:reset_odom __reset_odom) nil)
    )
   (send-super :init)
   (setq _reset_odom __reset_odom)
   self)
  (:reset_odom
   (&optional (__reset_odom :null))
   (if (not (eq __reset_odom :null)) (setq _reset_odom __reset_odom)) _reset_odom)
  (:serialization-length
   ()
   (+
    ;; bool _reset_odom
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _reset_odom
       (if _reset_odom (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _reset_odom
     (setq _reset_odom (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass rbo_create::ResetOdomResponse
  :super ros::object
  :slots (_success ))

(defmethod rbo_create::ResetOdomResponse
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

(defclass rbo_create::ResetOdom
  :super ros::object
  :slots ())

(setf (get rbo_create::ResetOdom :md5sum-) "d097c2ef742c9464ee3a0e656d7572a2")
(setf (get rbo_create::ResetOdom :datatype-) "rbo_create/ResetOdom")
(setf (get rbo_create::ResetOdom :request) rbo_create::ResetOdomRequest)
(setf (get rbo_create::ResetOdom :response) rbo_create::ResetOdomResponse)

(defmethod rbo_create::ResetOdomRequest
  (:response () (instance rbo_create::ResetOdomResponse :init)))

(setf (get rbo_create::ResetOdomRequest :md5sum-) "d097c2ef742c9464ee3a0e656d7572a2")
(setf (get rbo_create::ResetOdomRequest :datatype-) "rbo_create/ResetOdomRequest")
(setf (get rbo_create::ResetOdomRequest :definition-)
      "bool reset_odom
---
bool success

")

(setf (get rbo_create::ResetOdomResponse :md5sum-) "d097c2ef742c9464ee3a0e656d7572a2")
(setf (get rbo_create::ResetOdomResponse :datatype-) "rbo_create/ResetOdomResponse")
(setf (get rbo_create::ResetOdomResponse :definition-)
      "bool reset_odom
---
bool success

")



(provide :rbo_create/ResetOdom "d097c2ef742c9464ee3a0e656d7572a2")


