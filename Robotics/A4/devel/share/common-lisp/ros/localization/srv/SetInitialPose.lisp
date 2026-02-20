; Auto-generated. Do not edit!


(cl:in-package localization-srv)


;//! \htmlinclude SetInitialPose-request.msg.html

(cl:defclass <SetInitialPose-request> (roslisp-msg-protocol:ros-message)
  ((x
    :reader x
    :initarg :x
    :type cl:float
    :initform 0.0)
   (y
    :reader y
    :initarg :y
    :type cl:float
    :initform 0.0)
   (theta
    :reader theta
    :initarg :theta
    :type cl:float
    :initform 0.0)
   (var_x
    :reader var_x
    :initarg :var_x
    :type cl:float
    :initform 0.0)
   (var_y
    :reader var_y
    :initarg :var_y
    :type cl:float
    :initform 0.0)
   (var_theta
    :reader var_theta
    :initarg :var_theta
    :type cl:float
    :initform 0.0))
)

(cl:defclass SetInitialPose-request (<SetInitialPose-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetInitialPose-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetInitialPose-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name localization-srv:<SetInitialPose-request> is deprecated: use localization-srv:SetInitialPose-request instead.")))

(cl:ensure-generic-function 'x-val :lambda-list '(m))
(cl:defmethod x-val ((m <SetInitialPose-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader localization-srv:x-val is deprecated.  Use localization-srv:x instead.")
  (x m))

(cl:ensure-generic-function 'y-val :lambda-list '(m))
(cl:defmethod y-val ((m <SetInitialPose-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader localization-srv:y-val is deprecated.  Use localization-srv:y instead.")
  (y m))

(cl:ensure-generic-function 'theta-val :lambda-list '(m))
(cl:defmethod theta-val ((m <SetInitialPose-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader localization-srv:theta-val is deprecated.  Use localization-srv:theta instead.")
  (theta m))

(cl:ensure-generic-function 'var_x-val :lambda-list '(m))
(cl:defmethod var_x-val ((m <SetInitialPose-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader localization-srv:var_x-val is deprecated.  Use localization-srv:var_x instead.")
  (var_x m))

(cl:ensure-generic-function 'var_y-val :lambda-list '(m))
(cl:defmethod var_y-val ((m <SetInitialPose-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader localization-srv:var_y-val is deprecated.  Use localization-srv:var_y instead.")
  (var_y m))

(cl:ensure-generic-function 'var_theta-val :lambda-list '(m))
(cl:defmethod var_theta-val ((m <SetInitialPose-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader localization-srv:var_theta-val is deprecated.  Use localization-srv:var_theta instead.")
  (var_theta m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetInitialPose-request>) ostream)
  "Serializes a message object of type '<SetInitialPose-request>"
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'x))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'y))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'theta))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'var_x))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'var_y))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'var_theta))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetInitialPose-request>) istream)
  "Deserializes a message object of type '<SetInitialPose-request>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'x) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'y) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'theta) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'var_x) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'var_y) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'var_theta) (roslisp-utils:decode-double-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetInitialPose-request>)))
  "Returns string type for a service object of type '<SetInitialPose-request>"
  "localization/SetInitialPoseRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetInitialPose-request)))
  "Returns string type for a service object of type 'SetInitialPose-request"
  "localization/SetInitialPoseRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetInitialPose-request>)))
  "Returns md5sum for a message object of type '<SetInitialPose-request>"
  "2e51d06c4c070500696986776da1d81d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetInitialPose-request)))
  "Returns md5sum for a message object of type 'SetInitialPose-request"
  "2e51d06c4c070500696986776da1d81d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetInitialPose-request>)))
  "Returns full string definition for message of type '<SetInitialPose-request>"
  (cl:format cl:nil "float64 x~%float64 y~%float64 theta~%float64 var_x~%float64 var_y~%float64 var_theta~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetInitialPose-request)))
  "Returns full string definition for message of type 'SetInitialPose-request"
  (cl:format cl:nil "float64 x~%float64 y~%float64 theta~%float64 var_x~%float64 var_y~%float64 var_theta~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetInitialPose-request>))
  (cl:+ 0
     8
     8
     8
     8
     8
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetInitialPose-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetInitialPose-request
    (cl:cons ':x (x msg))
    (cl:cons ':y (y msg))
    (cl:cons ':theta (theta msg))
    (cl:cons ':var_x (var_x msg))
    (cl:cons ':var_y (var_y msg))
    (cl:cons ':var_theta (var_theta msg))
))
;//! \htmlinclude SetInitialPose-response.msg.html

(cl:defclass <SetInitialPose-response> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass SetInitialPose-response (<SetInitialPose-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetInitialPose-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetInitialPose-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name localization-srv:<SetInitialPose-response> is deprecated: use localization-srv:SetInitialPose-response instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetInitialPose-response>) ostream)
  "Serializes a message object of type '<SetInitialPose-response>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetInitialPose-response>) istream)
  "Deserializes a message object of type '<SetInitialPose-response>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetInitialPose-response>)))
  "Returns string type for a service object of type '<SetInitialPose-response>"
  "localization/SetInitialPoseResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetInitialPose-response)))
  "Returns string type for a service object of type 'SetInitialPose-response"
  "localization/SetInitialPoseResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetInitialPose-response>)))
  "Returns md5sum for a message object of type '<SetInitialPose-response>"
  "2e51d06c4c070500696986776da1d81d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetInitialPose-response)))
  "Returns md5sum for a message object of type 'SetInitialPose-response"
  "2e51d06c4c070500696986776da1d81d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetInitialPose-response>)))
  "Returns full string definition for message of type '<SetInitialPose-response>"
  (cl:format cl:nil "~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetInitialPose-response)))
  "Returns full string definition for message of type 'SetInitialPose-response"
  (cl:format cl:nil "~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetInitialPose-response>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetInitialPose-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetInitialPose-response
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetInitialPose)))
  'SetInitialPose-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetInitialPose)))
  'SetInitialPose-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetInitialPose)))
  "Returns string type for a service object of type '<SetInitialPose>"
  "localization/SetInitialPose")