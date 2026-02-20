;; Auto-generated. Do not edit!


(when (boundp 'localization::PoseWithWeightStamped)
  (if (not (find-package "LOCALIZATION"))
    (make-package "LOCALIZATION"))
  (shadow 'PoseWithWeightStamped (find-package "LOCALIZATION")))
(unless (find-package "LOCALIZATION::POSEWITHWEIGHTSTAMPED")
  (make-package "LOCALIZATION::POSEWITHWEIGHTSTAMPED"))

(in-package "ROS")
;;//! \htmlinclude PoseWithWeightStamped.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass localization::PoseWithWeightStamped
  :super ros::object
  :slots (_header _pose ))

(defmethod localization::PoseWithWeightStamped
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:pose __pose) (instance localization::PoseWithWeight :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _pose __pose)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:pose
   (&rest __pose)
   (if (keywordp (car __pose))
       (send* _pose __pose)
     (progn
       (if __pose (setq _pose (car __pose)))
       _pose)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; localization/PoseWithWeight _pose
    (send _pose :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; localization/PoseWithWeight _pose
       (send _pose :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; localization/PoseWithWeight _pose
     (send _pose :deserialize buf ptr-) (incf ptr- (send _pose :serialization-length))
   ;;
   self)
  )

(setf (get localization::PoseWithWeightStamped :md5sum-) "ab09b0858359112f2e340cfe7b7e9803")
(setf (get localization::PoseWithWeightStamped :datatype-) "localization/PoseWithWeightStamped")
(setf (get localization::PoseWithWeightStamped :definition-)
      "# This expresses an estimated pose with a reference coordinate frame and timestamp

Header header
localization/PoseWithWeight pose

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



(provide :localization/PoseWithWeightStamped "ab09b0858359112f2e340cfe7b7e9803")


