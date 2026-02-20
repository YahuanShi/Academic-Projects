;; Auto-generated. Do not edit!


(when (boundp 'localization::SetInitialPose)
  (if (not (find-package "LOCALIZATION"))
    (make-package "LOCALIZATION"))
  (shadow 'SetInitialPose (find-package "LOCALIZATION")))
(unless (find-package "LOCALIZATION::SETINITIALPOSE")
  (make-package "LOCALIZATION::SETINITIALPOSE"))
(unless (find-package "LOCALIZATION::SETINITIALPOSEREQUEST")
  (make-package "LOCALIZATION::SETINITIALPOSEREQUEST"))
(unless (find-package "LOCALIZATION::SETINITIALPOSERESPONSE")
  (make-package "LOCALIZATION::SETINITIALPOSERESPONSE"))

(in-package "ROS")





(defclass localization::SetInitialPoseRequest
  :super ros::object
  :slots (_x _y _theta _var_x _var_y _var_theta ))

(defmethod localization::SetInitialPoseRequest
  (:init
   (&key
    ((:x __x) 0.0)
    ((:y __y) 0.0)
    ((:theta __theta) 0.0)
    ((:var_x __var_x) 0.0)
    ((:var_y __var_y) 0.0)
    ((:var_theta __var_theta) 0.0)
    )
   (send-super :init)
   (setq _x (float __x))
   (setq _y (float __y))
   (setq _theta (float __theta))
   (setq _var_x (float __var_x))
   (setq _var_y (float __var_y))
   (setq _var_theta (float __var_theta))
   self)
  (:x
   (&optional __x)
   (if __x (setq _x __x)) _x)
  (:y
   (&optional __y)
   (if __y (setq _y __y)) _y)
  (:theta
   (&optional __theta)
   (if __theta (setq _theta __theta)) _theta)
  (:var_x
   (&optional __var_x)
   (if __var_x (setq _var_x __var_x)) _var_x)
  (:var_y
   (&optional __var_y)
   (if __var_y (setq _var_y __var_y)) _var_y)
  (:var_theta
   (&optional __var_theta)
   (if __var_theta (setq _var_theta __var_theta)) _var_theta)
  (:serialization-length
   ()
   (+
    ;; float64 _x
    8
    ;; float64 _y
    8
    ;; float64 _theta
    8
    ;; float64 _var_x
    8
    ;; float64 _var_y
    8
    ;; float64 _var_theta
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64 _x
       (sys::poke _x (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _y
       (sys::poke _y (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _theta
       (sys::poke _theta (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _var_x
       (sys::poke _var_x (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _var_y
       (sys::poke _var_y (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _var_theta
       (sys::poke _var_theta (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64 _x
     (setq _x (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _y
     (setq _y (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _theta
     (setq _theta (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _var_x
     (setq _var_x (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _var_y
     (setq _var_y (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _var_theta
     (setq _var_theta (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(defclass localization::SetInitialPoseResponse
  :super ros::object
  :slots ())

(defmethod localization::SetInitialPoseResponse
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass localization::SetInitialPose
  :super ros::object
  :slots ())

(setf (get localization::SetInitialPose :md5sum-) "2e51d06c4c070500696986776da1d81d")
(setf (get localization::SetInitialPose :datatype-) "localization/SetInitialPose")
(setf (get localization::SetInitialPose :request) localization::SetInitialPoseRequest)
(setf (get localization::SetInitialPose :response) localization::SetInitialPoseResponse)

(defmethod localization::SetInitialPoseRequest
  (:response () (instance localization::SetInitialPoseResponse :init)))

(setf (get localization::SetInitialPoseRequest :md5sum-) "2e51d06c4c070500696986776da1d81d")
(setf (get localization::SetInitialPoseRequest :datatype-) "localization/SetInitialPoseRequest")
(setf (get localization::SetInitialPoseRequest :definition-)
      "float64 x
float64 y
float64 theta
float64 var_x
float64 var_y
float64 var_theta
---

")

(setf (get localization::SetInitialPoseResponse :md5sum-) "2e51d06c4c070500696986776da1d81d")
(setf (get localization::SetInitialPoseResponse :datatype-) "localization/SetInitialPoseResponse")
(setf (get localization::SetInitialPoseResponse :definition-)
      "float64 x
float64 y
float64 theta
float64 var_x
float64 var_y
float64 var_theta
---

")



(provide :localization/SetInitialPose "2e51d06c4c070500696986776da1d81d")


