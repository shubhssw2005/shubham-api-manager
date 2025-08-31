import mongoose from 'mongoose';
const { Schema, models, model } = mongoose;




const RoleSchema = new Schema({
    name: {
        type: String,
        required: true,
        unique: true,
        lowercase: true,
        datatype: "textinput"
    },
    permission: {
        create: {
            type: Boolean,
            default: false,
            datatype: "toggleinput"
        },
        read: {
            type: Boolean,
            default: true,
            datatype: "toggleinput"
        },
        update: {
            type: Boolean,
            default: false,
            datatype: "toggleinput"
        },
        delete: {
            type: Boolean,
            default: false,
            datatype: "toggleinput"
        },
    },
    routes: {
        type: [String],
        default: [],
        datatype: "creatableselectmulti"
    },
    
    description: {
    type: String,
    datatype: "textarea",
    default: function(){return `${this.name} role`}
    },
    isSystemRole: {
        type: Boolean,
        default: false,
        datatype: "toggleinput"
    }
}, { timestamps: true});


RoleSchema.pre('save', function(next){
    if(this.name){
      this.name = this.name.toLowerCase();
    }
    next();
})

RoleSchema.virtual('displayName').get(function() {
    return this.name.charAt(0).toUpperCase() + this.name.slice(1);
})

RoleSchema.statics.findSystemRoles = function(){
    return this.find({isSystemRole: true});
}

export const Role = models.Role || model('Role', RoleSchema, 'roles');