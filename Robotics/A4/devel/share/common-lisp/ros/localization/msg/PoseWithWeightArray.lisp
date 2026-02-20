; Auto-generated. Do not edit!


(cl:in-package localization-msg)


;//! \htmlinclude PoseWithWeightArray.msg.html

(cl:defclass <PoseWithWeightArray> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (poses
    :reader poses
    :initarg :poses
    :type (cl:vector localization-msg:PoseWithWeight)
   :initform (cl:make-array 0 :element-type 'localization-msg:PoseWithWeight :initial-element (cl:make-instance 'localization-msg:PoseWithWeight))))
)

(cl:defclass PoseWithWeightArray (<PoseWithWeightArray>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PoseWithWeightArray>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PoseWithWeightArray)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name localization-msg:<PoseWithWeightArray> is deprecated: use localization-msg:PoseWithWeightArray instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <PoseWithWeightArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader localization-msg:header-val is deprecated.  Use localization-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'poses-val :lambda-list '(m))
(cl:defmethod poses-val ((m <PoseWithWeightArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader localization-msg:poses-val is deprecated.  Use localization-msg:poses instead.")
  (poses m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PoseWithWeightArray>) ostream)
  "Serializes a message object of type '<PoseWithWeightArray>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'poses))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'poses))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PoseWithWeightArray>) istream)
  "Deserializes a message object of type '<PoseWithWeightArray>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'poses) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'poses)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'localization-msg:PoseWithWeight))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PoseWithWeightArray>)))
  "Returns string type for a message object of type '<PoseWithWeightArray>"
  "localization/PoseWithWeightArray")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PoseWithWeightArray)))
  "Returns string type for a message object of type 'PoseWithWeightArray"
  "localization/PoseWithWeightArray")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PoseWithWeightArray>)))
  "Returns md5sum for a message object of type '<PoseWithWeightArray>"
  "0e902fceb5daae77c686f1ba0b414937")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PoseWithWeightArray)))
  "Returns md5sum for a message object of type 'PoseWithWeightArray"
  "0e902fceb5daae77c686f1ba0b414937")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PoseWithWeightArray>)))
  "Returns full string definition for message of type '<PoseWithWeightArray>"
  (cl:format cl:nil "# An array of poses with a header for global reference.~%~%Header header~%~%localization/PoseWithWeight[] poses~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: localization/PoseWithWeight~%# This represents a pose in free space with uncertainty.~%~%geometry_msgs/Pose pose~%~%# float weight~%float32 weight~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PoseWithWeightArray)))
  "Returns full string definition for message of type 'PoseWithWeightArray"
  (cl:format cl:nil "# An array of poses with a header for global reference.~%~%Header header~%~%localization/PoseWithWeight[] poses~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: localization/PoseWithWeight~%# This represents a pose in free space with uncertainty.~%~%geometry_msgs/Pose pose~%~%# float weight~%float32 weight~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PoseWithWeightArray>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'poses) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PoseWithWeightArray>))
  "Converts a ROS message object to a list"
  (cl:list 'PoseWithWeightArray
    (cl:cons ':header (header msg))
    (cl:cons ':poses (poses msg))
))
