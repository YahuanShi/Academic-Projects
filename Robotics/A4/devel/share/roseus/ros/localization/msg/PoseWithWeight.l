;; Auto-generated. Do not edit!


(when (boundp 'localization::PoseWithWeight)
  (if (not (find-package "LOCALIZATION"))
    (make-package "LOCALIZATION"))
  (shadow 'PoseWithWeight (find-package "LOCALIZATION")))
(unless (find-package "LOCALIZATION::POSEWITHWEIGHT")
  (make-package "LOCALIZATION::POSEWITHWEIGHT"))

(in-package "ROS")
;;//! \htmlinclude PoseWithWeight.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass localization::PoseWithWeight
  :super ros::object
  :slots (_pose _weight ))

(defmethod localization::PoseWithWeight
  (:init
   (&key
    ((:pose __pose) (instance geometry_msgs::Pose :init))
    ((:weight __weight) 0.0)
    )
   (send-super :init)
   (setq _pose __pose)
   (setq _weight (float __weight))
   self)
  (:pose
   (&rest __pose)
   (if (keywordp (car __pose))
       (send* _pose __pose)
     (progn
       (if __pose (setq _pose (car __pose)))
       _pose)))
  (:weight
   (&optional __weight)
   (if __weight (setq _weight __weight)) _weight)
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Pose _pose
    (send _pose :serialization-length)
    ;; float32 _weight
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Pose _pose
       (send _pose :serialize s)
     ;; float32 _weight
       (sys::poke _weight (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Pose _pose
     (send _pose :deserialize buf ptr-) (incf ptr- (send _pose :serialization-length))
   ;; float32 _weight
     (setq _weight (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get localization::PoseWithWeight :md5sum-) "f95056f54b0dad30bf874936f8a81e13")
(setf (get localization::PoseWithWeight :datatype-) "localization/PoseWithWeight")
(setf (get localization::PoseWithWeight :definition-)
      "# This represents a pose in free space with uncertainty.

geometry_msgs/Pose pose

# float weight
float32 weight

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

")



(provide :localization/PoseWithWeight "f95056f54b0dad30bf874936f8a81e13")


