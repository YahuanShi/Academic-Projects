; Auto-generated. Do not edit!


(cl:in-package rbo_create-srv)


;//! \htmlinclude ResetOdom-request.msg.html

(cl:defclass <ResetOdom-request> (roslisp-msg-protocol:ros-message)
  ((reset_odom
    :reader reset_odom
    :initarg :reset_odom
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass ResetOdom-request (<ResetOdom-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ResetOdom-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ResetOdom-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rbo_create-srv:<ResetOdom-request> is deprecated: use rbo_create-srv:ResetOdom-request instead.")))

(cl:ensure-generic-function 'reset_odom-val :lambda-list '(m))
(cl:defmethod reset_odom-val ((m <ResetOdom-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rbo_create-srv:reset_odom-val is deprecated.  Use rbo_create-srv:reset_odom instead.")
  (reset_odom m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ResetOdom-request>) ostream)
  "Serializes a message object of type '<ResetOdom-request>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'reset_odom) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ResetOdom-request>) istream)
  "Deserializes a message object of type '<ResetOdom-request>"
    (cl:setf (cl:slot-value msg 'reset_odom) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ResetOdom-request>)))
  "Returns string type for a service object of type '<ResetOdom-request>"
  "rbo_create/ResetOdomRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ResetOdom-request)))
  "Returns string type for a service object of type 'ResetOdom-request"
  "rbo_create/ResetOdomRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ResetOdom-request>)))
  "Returns md5sum for a message object of type '<ResetOdom-request>"
  "d097c2ef742c9464ee3a0e656d7572a2")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ResetOdom-request)))
  "Returns md5sum for a message object of type 'ResetOdom-request"
  "d097c2ef742c9464ee3a0e656d7572a2")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ResetOdom-request>)))
  "Returns full string definition for message of type '<ResetOdom-request>"
  (cl:format cl:nil "bool reset_odom~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ResetOdom-request)))
  "Returns full string definition for message of type 'ResetOdom-request"
  (cl:format cl:nil "bool reset_odom~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ResetOdom-request>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ResetOdom-request>))
  "Converts a ROS message object to a list"
  (cl:list 'ResetOdom-request
    (cl:cons ':reset_odom (reset_odom msg))
))
;//! \htmlinclude ResetOdom-response.msg.html

(cl:defclass <ResetOdom-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass ResetOdom-response (<ResetOdom-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ResetOdom-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ResetOdom-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rbo_create-srv:<ResetOdom-response> is deprecated: use rbo_create-srv:ResetOdom-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <ResetOdom-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rbo_create-srv:success-val is deprecated.  Use rbo_create-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ResetOdom-response>) ostream)
  "Serializes a message object of type '<ResetOdom-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ResetOdom-response>) istream)
  "Deserializes a message object of type '<ResetOdom-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ResetOdom-response>)))
  "Returns string type for a service object of type '<ResetOdom-response>"
  "rbo_create/ResetOdomResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ResetOdom-response)))
  "Returns string type for a service object of type 'ResetOdom-response"
  "rbo_create/ResetOdomResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ResetOdom-response>)))
  "Returns md5sum for a message object of type '<ResetOdom-response>"
  "d097c2ef742c9464ee3a0e656d7572a2")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ResetOdom-response)))
  "Returns md5sum for a message object of type 'ResetOdom-response"
  "d097c2ef742c9464ee3a0e656d7572a2")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ResetOdom-response>)))
  "Returns full string definition for message of type '<ResetOdom-response>"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ResetOdom-response)))
  "Returns full string definition for message of type 'ResetOdom-response"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ResetOdom-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ResetOdom-response>))
  "Converts a ROS message object to a list"
  (cl:list 'ResetOdom-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'ResetOdom)))
  'ResetOdom-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'ResetOdom)))
  'ResetOdom-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ResetOdom)))
  "Returns string type for a service object of type '<ResetOdom>"
  "rbo_create/ResetOdom")