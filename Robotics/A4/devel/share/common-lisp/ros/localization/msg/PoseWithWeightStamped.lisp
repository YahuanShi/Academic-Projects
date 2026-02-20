; Auto-generated. Do not edit!


(cl:in-package localization-msg)


;//! \htmlinclude PoseWithWeightStamped.msg.html

(cl:defclass <PoseWithWeightStamped> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (pose
    :reader pose
    :initarg :pose
    :type localization-msg:PoseWithWeight
    :initform (cl:make-instance 'localization-msg:PoseWithWeight)))
)

(cl:defclass PoseWithWeightStamped (<PoseWithWeightStamped>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PoseWithWeightStamped>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PoseWithWeightStamped)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name localization-msg:<PoseWithWeightStamped> is deprecated: use localization-msg:PoseWithWeightStamped instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <PoseWithWeightStamped>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader localization-msg:header-val is deprecated.  Use localization-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'pose-val :lambda-list '(m))
(cl:defmethod pose-val ((m <PoseWithWeightStamped>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader localization-msg:pose-val is deprecated.  Use localization-msg:pose instead.")
  (pose m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PoseWithWeightStamped>) ostream)
  "Serializes a message object of type '<PoseWithWeightStamped>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'pose) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PoseWithWeightStamped>) istream)
  "Deserializes a message object of type '<PoseWithWeightStamped>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'pose) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PoseWithWeightStamped>)))
  "Returns string type for a message object of type '<PoseWithWeightStamped>"
  "localization/PoseWithWeightStamped")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PoseWithWeightStamped)))
  "Returns string type for a message object of type 'PoseWithWeightStamped"
  "localization/PoseWithWeightStamped")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PoseWithWeightStamped>)))
  "Returns md5sum for a message object of type '<PoseWithWeightStamped>"
  "ab09b0858359112f2e340cfe7b7e9803")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PoseWithWeightStamped)))
  "Returns md5sum for a message object of type 'PoseWithWeightStamped"
  "ab09b0858359112f2e340cfe7b7e9803")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PoseWithWeightStamped>)))
  "Returns full string definition for message of type '<PoseWithWeightStamped>"
  (cl:format cl:nil "# This expresses an estimated pose with a reference coordinate frame and timestamp~%~%Header header~%localization/PoseWithWeight pose~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: localization/PoseWithWeight~%# This represents a pose in free space with uncertainty.~%~%geometry_msgs/Pose pose~%~%# float weight~%float32 weight~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PoseWithWeightStamped)))
  "Returns full string definition for message of type 'PoseWithWeightStamped"
  (cl:format cl:nil "# This expresses an estimated pose with a reference coordinate frame and timestamp~%~%Header header~%localization/PoseWithWeight pose~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: localization/PoseWithWeight~%# This represents a pose in free space with uncertainty.~%~%geometry_msgs/Pose pose~%~%# float weight~%float32 weight~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PoseWithWeightStamped>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'pose))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PoseWithWeightStamped>))
  "Converts a ROS message object to a list"
  (cl:list 'PoseWithWeightStamped
    (cl:cons ':header (header msg))
    (cl:cons ':pose (pose msg))
))
