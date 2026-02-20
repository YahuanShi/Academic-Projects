;; Auto-generated. Do not edit!


(when (boundp 'localization::PoseWithWeightArray)
  (if (not (find-package "LOCALIZATION"))
    (make-package "LOCALIZATION"))
  (shadow 'PoseWithWeightArray (find-package "LOCALIZATION")))
(unless (find-package "LOCALIZATION::POSEWITHWEIGHTARRAY")
  (make-package "LOCALIZATION::POSEWITHWEIGHTARRAY"))

(in-package "ROS")
;;//! \htmlinclude PoseWithWeightArray.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass localization::PoseWithWeightArray
  :super ros::object
  :slots (_header _poses ))

(defmethod localization::PoseWithWeightArray
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:poses __poses) ())
    )
   (send-super :init)
   (setq _header __header)
   (setq _poses __poses)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:poses
   (&rest __poses)
   (if (keywordp (car __poses))
       (send* _poses __poses)
     (progn
       (if __poses (setq _poses (car __poses)))
       _poses)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; localization/PoseWithWeight[] _poses
    (apply #'+ (send-all _poses :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; localization/PoseWithWeight[] _poses
     (write-long (length _poses) s)
     (dolist (elem _poses)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; localization/PoseWithWeight[] _poses
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _poses (let (r) (dotimes (i n) (push (instance localization::PoseWithWeight :init) r)) r))
     (dolist (elem- _poses)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get localization::PoseWithWeightArray :md5sum-) "0e902fceb5daae77c686f1ba0b414937")
(setf (get localization::PoseWithWeightArray :datatype-) "localization/PoseWithWeightArray")
(setf (get localization::PoseWithWeightArray :definition-)
      "# An array of poses with a header for global reference.

Header header

localization/PoseWithWeight[] poses

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: localization/PoseWithWeight
# This represents a pose in free space with uncertainty.

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



(provide :localization/PoseWithWeightArray "0e902fceb5daae77c686f1ba0b414937")


