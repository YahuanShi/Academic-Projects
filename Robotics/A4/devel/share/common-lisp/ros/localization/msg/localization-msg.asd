
(cl:in-package :asdf)

(defsystem "localization-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "PoseWithWeight" :depends-on ("_package_PoseWithWeight"))
    (:file "_package_PoseWithWeight" :depends-on ("_package"))
    (:file "PoseWithWeightArray" :depends-on ("_package_PoseWithWeightArray"))
    (:file "_package_PoseWithWeightArray" :depends-on ("_package"))
    (:file "PoseWithWeightStamped" :depends-on ("_package_PoseWithWeightStamped"))
    (:file "_package_PoseWithWeightStamped" :depends-on ("_package"))
  ))