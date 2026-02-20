; Auto-generated. Do not edit!


(cl:in-package localization-msg)


;//! \htmlinclude PoseWithWeight.msg.html

(cl:defclass <PoseWithWeight> (roslisp-msg-protocol:ros-message)
  ((pose
    :reader pose
    :initarg :pose
    :type geometry_msgs-msg:Pose
    :initform (cl:make-instance 'geometry_msgs-msg:Pose))
   (weight
    :reader weight
    :initarg :weight
    :type cl:float
    :initform 0.0))
)

(cl:defclass PoseWithWeight (<PoseWithWeight>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PoseWithWeight>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PoseWithWeight)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name localization-msg:<PoseWithWeight> is deprecated: use localization-msg:PoseWithWeight instead.")))

(cl:ensure-generic-function 'pose-val :lambda-list '(m))
(cl:defmethod pose-val ((m <PoseWithWeight>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader localization-msg:pose-val is deprecated.  Use localization-msg:pose instead.")
  (pose m))

(cl:ensure-generic-function 'weight-val :lambda-list '(m))
(cl:defmethod weight-val ((m <PoseWithWeight>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader localization-msg:weight-val is deprecated.  Use localization-msg:weight instead.")
  (weight m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PoseWithWeight>) ostream)
  "Serializes a message object of type '<PoseWithWeight>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'pose) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'weight))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PoseWithWeight>) istream)
  "Deserializes a message object of type '<PoseWithWeight>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'pose) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'weight) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PoseWithWeight>)))
  "Returns string type for a message object of type '<PoseWithWeight>"
  "localization/PoseWithWeight")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PoseWithWeight)))
  "Returns string type for a message object of type 'PoseWithWeight"
  "localization/PoseWithWeight")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PoseWithWeight>)))
  "Returns md5sum for a message object of type '<PoseWithWeight>"
  "f95056f54b0dad30bf874936f8a81e13")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PoseWithWeight)))
  "Returns md5sum for a message object of type 'PoseWithWeight"
  "f95056f54b0dad30bf874936f8a81e13")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PoseWithWeight>)))
  "Returns full string definition for message of type '<PoseWithWeight>"
  (cl:format cl:nil "# This represents a pose in free space with uncertainty.~%~%geometry_msgs/Pose pose~%~%# float weight~%float32 weight~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PoseWithWeight)))
  "Returns full string definition for message of type 'PoseWithWeight"
  (cl:format cl:nil "# This represents a pose in free space with uncertainty.~%~%geometry_msgs/Pose pose~%~%# float weight~%float32 weight~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PoseWithWeight>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'pose))
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PoseWithWeight>))
  "Converts a ROS message object to a list"
  (cl:list 'PoseWithWeight
    (cl:cons ':pose (pose msg))
    (cl:cons ':weight (weight msg))
))
