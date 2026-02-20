
(cl:in-package :asdf)

(defsystem "rbo_create-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "Leds" :depends-on ("_package_Leds"))
    (:file "_package_Leds" :depends-on ("_package"))
    (:file "ResetOdom" :depends-on ("_package_ResetOdom"))
    (:file "_package_ResetOdom" :depends-on ("_package"))
    (:file "Tank" :depends-on ("_package_Tank"))
    (:file "_package_Tank" :depends-on ("_package"))
  ))