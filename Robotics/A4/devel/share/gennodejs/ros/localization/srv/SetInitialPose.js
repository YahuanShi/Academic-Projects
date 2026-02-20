// Auto-generated. Do not edit!

// (in-package localization.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class SetInitialPoseRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.x = null;
      this.y = null;
      this.theta = null;
      this.var_x = null;
      this.var_y = null;
      this.var_theta = null;
    }
    else {
      if (initObj.hasOwnProperty('x')) {
        this.x = initObj.x
      }
      else {
        this.x = 0.0;
      }
      if (initObj.hasOwnProperty('y')) {
        this.y = initObj.y
      }
      else {
        this.y = 0.0;
      }
      if (initObj.hasOwnProperty('theta')) {
        this.theta = initObj.theta
      }
      else {
        this.theta = 0.0;
      }
      if (initObj.hasOwnProperty('var_x')) {
        this.var_x = initObj.var_x
      }
      else {
        this.var_x = 0.0;
      }
      if (initObj.hasOwnProperty('var_y')) {
        this.var_y = initObj.var_y
      }
      else {
        this.var_y = 0.0;
      }
      if (initObj.hasOwnProperty('var_theta')) {
        this.var_theta = initObj.var_theta
      }
      else {
        this.var_theta = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetInitialPoseRequest
    // Serialize message field [x]
    bufferOffset = _serializer.float64(obj.x, buffer, bufferOffset);
    // Serialize message field [y]
    bufferOffset = _serializer.float64(obj.y, buffer, bufferOffset);
    // Serialize message field [theta]
    bufferOffset = _serializer.float64(obj.theta, buffer, bufferOffset);
    // Serialize message field [var_x]
    bufferOffset = _serializer.float64(obj.var_x, buffer, bufferOffset);
    // Serialize message field [var_y]
    bufferOffset = _serializer.float64(obj.var_y, buffer, bufferOffset);
    // Serialize message field [var_theta]
    bufferOffset = _serializer.float64(obj.var_theta, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetInitialPoseRequest
    let len;
    let data = new SetInitialPoseRequest(null);
    // Deserialize message field [x]
    data.x = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [y]
    data.y = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [theta]
    data.theta = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [var_x]
    data.var_x = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [var_y]
    data.var_y = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [var_theta]
    data.var_theta = _deserializer.float64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 48;
  }

  static datatype() {
    // Returns string type for a service object
    return 'localization/SetInitialPoseRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '2e51d06c4c070500696986776da1d81d';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float64 x
    float64 y
    float64 theta
    float64 var_x
    float64 var_y
    float64 var_theta
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetInitialPoseRequest(null);
    if (msg.x !== undefined) {
      resolved.x = msg.x;
    }
    else {
      resolved.x = 0.0
    }

    if (msg.y !== undefined) {
      resolved.y = msg.y;
    }
    else {
      resolved.y = 0.0
    }

    if (msg.theta !== undefined) {
      resolved.theta = msg.theta;
    }
    else {
      resolved.theta = 0.0
    }

    if (msg.var_x !== undefined) {
      resolved.var_x = msg.var_x;
    }
    else {
      resolved.var_x = 0.0
    }

    if (msg.var_y !== undefined) {
      resolved.var_y = msg.var_y;
    }
    else {
      resolved.var_y = 0.0
    }

    if (msg.var_theta !== undefined) {
      resolved.var_theta = msg.var_theta;
    }
    else {
      resolved.var_theta = 0.0
    }

    return resolved;
    }
};

class SetInitialPoseResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetInitialPoseResponse
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetInitialPoseResponse
    let len;
    let data = new SetInitialPoseResponse(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'localization/SetInitialPoseResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd41d8cd98f00b204e9800998ecf8427e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetInitialPoseResponse(null);
    return resolved;
    }
};

module.exports = {
  Request: SetInitialPoseRequest,
  Response: SetInitialPoseResponse,
  md5sum() { return '2e51d06c4c070500696986776da1d81d'; },
  datatype() { return 'localization/SetInitialPose'; }
};
