(define (stream ltamp)

	; External predicates
	(:predicate (ControlPoseCollision ?arm ?control ?obj ?pose)
	    (and (IsControl ?arm ?control) (IsPose ?obj ?pose) (Movable ?obj)))
	(:predicate (ControlConfCollision ?arm ?control ?arm2 ?conf2)
	    (and (IsControl ?arm ?control) (IsConf ?arm2 ?conf2)))

	;(:predicate (PoseCollision ?obj ?pose ?obj2 ?pose2)
	;	(and (IsPose ?obj ?pose) (IsPose ?obj2 ?pose2))
	;)
    (:predicate (ConfConfCollision ?arm ?conf ?arm2 ?conf2)
	    (and (IsConf ?arm ?conf) (IsConf ?arm2 ?conf2)))
    ; TODO: holding/arm predicate
	;(:predicate (WillSpill ?arm ?conf ?obj ?grasp)
	;	(and (IsConf ?arm ?conf) (IsGrasp ?grasp))
	;)

	; TODO: combine these
    ;(:predicate (ControlGraspCollision ?arm ?control ?arm2 ?conf2 ?obj ?grasp)
    ;    (and (IsControl ?arm ?control) (IsConf ?arm2 ?conf2) (IsGrasp ?obj ?grasp)))
    ;(:predicate (GraspGraspCollision ...))

	; Streams
	(:stream sample-motion
		:inputs (?arm ?conf1 ?conf2)
		:domain (and (IsConf ?arm ?conf1) (IsConf ?arm ?conf2)
					 (IsArm ?arm))
		:fluents (AtPose AtConf AtGrasp Contains)
		:outputs (?control)
		:certified (IsMove ?arm ?conf1 ?conf2 ?control) ; No need for a control type
	)

    ;(:stream sample-grasp
	;	:inputs (?obj)
	;	:domain (Graspable ?obj)
	;	:outputs (?grasp)
	;	:certified (IsGrasp ?obj ?grasp)
	;)

	(:stream test-reachable
		:inputs (?arm ?obj ?pose)
		:domain (and (IsArm ?arm) (IsPose ?obj ?pose))
		:outputs ()
		:certified (Reachable ?arm ?obj ?pose)
    )

	;(:stream sample-pick
	;	:inputs (?arm ?obj ?pose ?grasp)
	;	:domain (and (IsArm ?arm) (IsPose ?obj ?pose) (IsGrasp ?obj ?grasp))
	;	:outputs (?conf ?conf2 ?control)
	;	:certified (and (IsPick ?arm ?obj ?pose ?grasp ?conf ?conf2 ?control)
	;	                (IsConf ?arm ?conf) (IsConf ?arm ?conf2) (IsControl ?arm ?control))
	;)
	(:stream sample-pick
		:inputs (?arm ?obj ?pose)
		:domain (and (IsArm ?arm) (IsPose ?obj ?pose) (Graspable ?obj))
		:outputs (?grasp ?conf ?conf2 ?control)
		:certified (and (IsPick ?arm ?obj ?pose ?grasp ?conf ?conf2 ?control)
		                (IsGrasp ?obj ?grasp) (IsConf ?arm ?conf) (IsConf ?arm ?conf2) (IsControl ?arm ?control))
	)

    ; TODO: stashed combination of sample-pose and sample-place
	(:stream sample-pose
		:inputs (?obj ?surface ?pose2 ?link)
		:domain (and (Stackable ?obj ?surface ?link) (IsPose ?surface ?pose2))
		:outputs (?pose)
		:certified (and (IsSupported ?obj ?pose ?surface ?pose2 ?link)
		                (IsPose ?obj ?pose))
	)
	(:stream sample-place
		:inputs (?arm ?obj ?pose ?grasp)
		:domain (and (IsArm ?arm) (IsPose ?obj ?pose) (IsGrasp ?obj ?grasp))
		:outputs (?conf1 ?conf2 ?control)
		:certified (and (IsPlace ?arm ?obj ?pose ?grasp ?conf1 ?conf2 ?control)
						(IsConf ?arm ?conf1) (IsConf ?arm ?conf2) (IsControl ?arm ?control))
	)

	(:stream sample-pour
		:inputs (?arm ?bowl ?pose ?cup ?grasp) ; TODO: can pour ?bowl & ?cup
		:domain (and (IsArm ?arm) (IsPose ?bowl ?pose) (IsGrasp ?cup ?grasp) (CanHold ?bowl) (Pourable ?cup))
		:outputs (?conf1 ?conf2 ?control)
		:certified (and (IsPour ?arm ?bowl ?pose ?cup ?grasp ?conf1 ?conf2 ?control)
						(IsConf ?arm ?conf1) (IsConf ?arm ?conf2) (IsControl ?arm ?control))
	)

	(:stream sample-stir
		:inputs (?arm ?bowl ?pose ?stirrer ?grasp)
		:domain (and (IsArm ?arm) (IsPose ?bowl ?pose) (IsGrasp ?stirrer ?grasp) (CanHold ?bowl) (Stirrable ?stirrer))
		:outputs (?conf1 ?conf2 ?control)
		:certified (and (IsStir ?arm ?bowl ?pose ?stirrer ?grasp ?conf1 ?conf2 ?control)
						(IsConf ?arm ?conf1) (IsConf ?arm ?conf2) (IsControl ?arm ?control))
	)

	(:stream sample-scoop
		:inputs (?arm ?bowl ?pose ?spoon ?grasp)
		:domain (and (IsArm ?arm) (IsPose ?bowl ?pose) (IsGrasp ?spoon ?grasp) (Scoopable ?bowl) (CanScoop ?spoon))
		:outputs (?conf1 ?conf2 ?control)
		:certified (and (IsScoop ?arm ?bowl ?pose ?spoon ?grasp ?conf1 ?conf2 ?control)
						(IsConf ?arm ?conf1) (IsConf ?arm ?conf2) (IsControl ?arm ?control))
	)

    ; TODO(caelan): indicate which pairs of surfaces can be pushed between per object (e.g. obj: table->edge)
    ; TODO(caelan): push on a surface w/ supported
    ; TODO(caelan): unify Contained & IsSupported
	(:stream sample-push
		:inputs (?arm ?obj ?pose1 ?region)
		:domain (and (IsArm ?arm) (IsPose ?obj ?pose1) (CanPush ?obj ?region))
		:outputs (?pose2 ?conf1 ?conf2 ?control)
		:certified (and (IsPush ?arm ?obj ?pose1 ?pose2 ?conf1 ?conf2 ?control) (Contained ?obj ?pose2 ?region)
						(IsPose ?obj ?pose2) (IsConf ?arm ?conf1) (IsConf ?arm ?conf2) (IsControl ?arm ?control))
	)

    (:stream sample-press
		:inputs (?arm ?button)
		:domain (and (IsArm ?arm) (Pressable ?button))
		:outputs (?conf1 ?conf2 ?control)
		:certified (and (IsPress ?arm ?button ?conf1 ?conf2 ?control)
						(IsConf ?arm ?conf1) (IsConf ?arm ?conf2) (IsControl ?arm ?control))
	)
)
