
(cl:in-package :asdf)

(defsystem "localization-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "SetInitialPose" :depends-on ("_package_SetInitialPose"))
    (:file "_package_SetInitialPose" :depends-on ("_package"))
  ))